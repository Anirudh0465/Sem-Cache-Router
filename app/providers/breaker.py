# Per provider circuit breaker.
#
# Will hold:
#   a BreakerState enum of CLOSED, OPEN and HALF_OPEN
#   a CircuitBreaker class with call, _on_success, _on_failure, _backoff and a
#   helper that decides whether an exception counts as a failure
#
# State transitions:
#   CLOSED     passes traffic and counts failures
#   OPEN       rejects immediately, without attempting a call
#   HALF_OPEN  allows a limited number of probes after the backoff expires
#   a probe that succeeds closes the breaker
#   a probe that fails reopens it with a longer backoff
#
# Backoff shape:
#   delay  = min(base * 2 ** attempt, ceiling)
#   actual = random.uniform(0, delay)
#
# Why jitter: without it, every client that failed at the same moment retries at
# the same moment, and a recovering provider receives a synchronised burst
# precisely when it is least able to absorb one. Jitter turns a thundering herd
# into a gradual ramp.
#
# Failure classification is deliberate. Timeouts, 5xx responses and provider
# 429s count toward the breaker. A 400 does not, because a malformed request is
# a caller defect, and counting it would let one badly behaved client trip
# failover for everybody.

from __future__ import annotations

import asyncio
import enum
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class BreakerState(enum.Enum):
    """The three states a circuit breaker moves through.

    CLOSED is the normal operating state. OPEN rejects immediately. HALF_OPEN
    allows a limited probe to decide whether to close or reopen.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the breaker is OPEN.

    Callers should catch this and route to a failover provider rather than
    surfacing it to the end user.
    """


class ProviderError(Exception):
    """Wraps a provider HTTP error so the breaker can inspect the status code.

    Provider clients raise this with the HTTP status code so that the breaker's
    failure classification can distinguish a 400 (caller defect, does not count)
    from a 5xx or 429 (provider problem, counts toward the threshold).
    """

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or f"Provider returned {status_code}")


class CircuitBreaker:
    """Per-provider circuit breaker with exponential backoff and jitter.

    All four tunables come from settings, not constants in this module:
      failure_threshold  – failures needed to trip the breaker open
      window_seconds     – the rolling window those failures are counted in
      base_backoff       – starting retry delay in seconds
      backoff_ceiling    – the cap that exponential growth stops at

    The guarded call is async. Failure classification is deliberate: timeouts,
    5xx responses and provider 429s count. A 400 does not.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        base_backoff: float = 1.0,
        backoff_ceiling: float = 60.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds
        self._base_backoff = base_backoff
        self._backoff_ceiling = backoff_ceiling

        # Injectable clock for testing. Avoids sleeping through real backoff
        # intervals in the test suite.
        self._clock = clock or time.monotonic

        self._state = BreakerState.CLOSED
        # Timestamps of failures within the current window, oldest first.
        self._failures: deque[float] = deque()
        # How many times the breaker has transitioned to OPEN without a
        # successful close in between. Drives the exponential backoff.
        self._attempt = 0
        # The wall-clock time at which the current OPEN period expires and the
        # breaker moves to HALF_OPEN.
        self._open_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        """Return the current breaker state.

        If the breaker is OPEN and the backoff has expired, this transparently
        transitions to HALF_OPEN so that the next call acts as a probe.
        """
        if self._state is BreakerState.OPEN and self._clock() >= self._open_until:
            self._state = BreakerState.HALF_OPEN
        return self._state

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute *func* through the breaker.

        In CLOSED state, the call proceeds normally and failures are tracked.
        In OPEN state, the call is rejected immediately with CircuitBreakerOpen.
        In HALF_OPEN state, the call acts as a probe: success closes the
        breaker, failure reopens it with a longer backoff.

        Raises CircuitBreakerOpen if the breaker is OPEN and the backoff has not
        yet expired.
        """
        current = self.state

        if current is BreakerState.OPEN:
            raise CircuitBreakerOpen(
                f"Breaker is open until {self._open_until:.1f}"
            )

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            if self._is_failure(exc):
                self._on_failure()
            raise
        else:
            self._on_success()
            return result

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        """Record a successful call.

        In HALF_OPEN state a success closes the breaker and resets the attempt
        counter, because the provider has demonstrated recovery. In CLOSED
        state a success has no effect on failure tracking; failures age out of
        the window naturally.
        """
        if self._state is BreakerState.HALF_OPEN:
            self._state = BreakerState.CLOSED
            self._attempt = 0
            self._failures.clear()

    def _on_failure(self) -> None:
        """Record a failed call and potentially trip the breaker.

        In CLOSED state the failure is appended to the rolling window. If the
        window contains at least *failure_threshold* entries, the breaker opens.

        In HALF_OPEN state any failure immediately reopens the breaker with an
        incremented attempt counter, producing a longer backoff. This prevents
        a still-failing provider from receiving a steady stream of probes.
        """
        now = self._clock()

        if self._state is BreakerState.HALF_OPEN:
            self._attempt += 1
            self._trip(now)
            return

        # CLOSED: record the failure and prune entries outside the window.
        self._failures.append(now)
        cutoff = now - self._window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

        if len(self._failures) >= self._failure_threshold:
            self._attempt += 1
            self._trip(now)

    def _trip(self, now: float) -> None:
        """Move the breaker to OPEN with a jittered backoff.

        The backoff formula:
            delay  = min(base * 2 ** attempt, ceiling)
            actual = random.uniform(0, delay)

        Jitter is the entire point. Without it every client that failed at the
        same moment retries at the same moment, and a recovering provider gets a
        synchronised burst precisely when it can least absorb one.
        """
        delay = min(
            self._base_backoff * (2 ** self._attempt),
            self._backoff_ceiling,
        )
        actual = random.uniform(0, delay)
        self._open_until = now + actual
        self._state = BreakerState.OPEN
        self._failures.clear()

    # ------------------------------------------------------------------
    # Failure classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_failure(exc: Exception) -> bool:
        """Decide whether *exc* counts toward the breaker threshold.

        Timeouts and 5xx and 429 responses are provider problems and count.
        A 400 is a caller defect and does not count, because one badly behaved
        client should not trip failover for everybody.

        The check inspects a ``status_code`` attribute if present, so it works
        with ProviderError and with any SDK exception that exposes one.
        """
        # Timeouts always count.
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True

        status: int | None = getattr(exc, "status_code", None)
        if status is not None:
            # 5xx: server error on the provider side.
            if 500 <= status < 600:
                return True
            # 429: provider rate limit, distinct from our own admission
            # control. Everything else, including 400, is not a provider
            # failure.
            return status == 429

        # Unknown exceptions without a status code are treated as failures,
        # because the call did not succeed and we cannot classify it otherwise.
        return True
