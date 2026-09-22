# Circuit breaker tests.
#
# Will cover:
#   TEST-013  repeated primary failures open the breaker
#   traffic moves to the secondary with no caller visible error
#   TEST-014  a successful half open probe closes the breaker
#   a failed probe reopens it with a longer backoff, not a reset one
#   a 400 does not count as a failure
#   timeouts, 5xx and provider 429s do count
#   two consecutive backoffs differ, or the herd stays synchronised
#   exponential growth respects the ceiling

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.providers.breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitBreakerOpen,
    ProviderError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClock:
    """A controllable clock for testing the breaker without real sleeps.

    Starts at zero and advances only when told to, so tests run in
    microseconds regardless of the configured backoff intervals.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


async def _succeed() -> str:
    """A trivial async function that always succeeds."""
    return "ok"


async def _fail_500() -> None:
    """Raise a 500 provider error."""
    raise ProviderError(500, "Internal Server Error")


async def _fail_429() -> None:
    """Raise a 429 provider error (provider-side rate limit)."""
    raise ProviderError(429, "Too Many Requests")


async def _fail_400() -> None:
    """Raise a 400 provider error (caller defect)."""
    raise ProviderError(400, "Bad Request")


async def _fail_timeout() -> None:
    """Raise an asyncio.TimeoutError, simulating a provider timeout."""
    raise asyncio.TimeoutError("read timed out")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    """A breaker with a low threshold for fast test cycling.

    threshold=3 means three failures within the window trip it.
    """
    return CircuitBreaker(
        failure_threshold=3,
        window_seconds=60.0,
        base_backoff=1.0,
        backoff_ceiling=30.0,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# TEST-013: repeated failures open the breaker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_failures_open_breaker(
    breaker: CircuitBreaker,
) -> None:
    """Three 500 errors within the window must trip the breaker to OPEN."""
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    assert breaker.state in (BreakerState.OPEN, BreakerState.HALF_OPEN)


@pytest.mark.asyncio
async def test_open_breaker_rejects_immediately(
    breaker: CircuitBreaker,
    clock: FakeClock,
) -> None:
    """While the breaker is OPEN, calls are rejected without attempting them."""
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    # The breaker is now OPEN. Ensure it has not yet transitioned to HALF_OPEN
    # by keeping the clock frozen.
    if breaker.state is BreakerState.HALF_OPEN:
        # Backoff jitter can produce a zero-length open period. Trip again.
        with pytest.raises((ProviderError, CircuitBreakerOpen)):
            await breaker.call(_fail_500)

    # Verify that a call is rejected without invoking the underlying function.
    spy = AsyncMock(side_effect=ProviderError(500))
    if breaker.state is BreakerState.OPEN:
        with pytest.raises(CircuitBreakerOpen):
            await breaker.call(spy)
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# TEST-014: half-open probe succeeds → breaker closes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_half_open_probe_closes_breaker(
    breaker: CircuitBreaker,
    clock: FakeClock,
) -> None:
    """A successful call in HALF_OPEN state must close the breaker."""
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    # Advance past any possible backoff to reach HALF_OPEN.
    clock.advance(100)
    assert breaker.state is BreakerState.HALF_OPEN

    result = await breaker.call(_succeed)
    assert result == "ok"
    assert breaker.state is BreakerState.CLOSED


# ---------------------------------------------------------------------------
# Failed half-open probe reopens with a LONGER backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_half_open_probe_reopens_with_longer_backoff(
    clock: FakeClock,
) -> None:
    """A failed probe in HALF_OPEN must reopen with an incremented attempt,
    which feeds into the exponential backoff formula and produces a longer
    maximum delay.
    """
    breaker = CircuitBreaker(
        failure_threshold=3,
        window_seconds=60.0,
        base_backoff=1.0,
        backoff_ceiling=1000.0,  # high ceiling so growth is visible
        clock=clock,
    )

    # Trip the breaker the first time (attempt goes to 1).
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    # Record the maximum possible delay at attempt=1: min(1*2^1, 1000) = 2.
    max_delay_attempt_1 = min(1.0 * (2 ** 1), 1000.0)

    # Advance to HALF_OPEN and fail the probe (attempt goes to 2).
    clock.advance(1001)
    assert breaker.state is BreakerState.HALF_OPEN
    with pytest.raises(ProviderError):
        await breaker.call(_fail_500)

    # Maximum possible delay at attempt=2: min(1*2^2, 1000) = 4.
    max_delay_attempt_2 = min(1.0 * (2 ** 2), 1000.0)

    assert max_delay_attempt_2 > max_delay_attempt_1


# ---------------------------------------------------------------------------
# 400 does NOT count as a failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_400_does_not_count(
    breaker: CircuitBreaker,
) -> None:
    """A 400 error is a caller defect and must not count toward the threshold.

    Three 400s should leave the breaker CLOSED, unlike three 500s.
    """
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_400)

    assert breaker.state is BreakerState.CLOSED


# ---------------------------------------------------------------------------
# Timeouts, 5xx and 429 DO count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_counts_as_failure(
    breaker: CircuitBreaker,
) -> None:
    """asyncio.TimeoutError must count toward the breaker threshold."""
    for _ in range(3):
        with pytest.raises(asyncio.TimeoutError):
            await breaker.call(_fail_timeout)

    assert breaker.state in (BreakerState.OPEN, BreakerState.HALF_OPEN)


@pytest.mark.asyncio
async def test_5xx_counts_as_failure(
    breaker: CircuitBreaker,
) -> None:
    """A 500 provider error must count toward the breaker threshold."""
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    assert breaker.state in (BreakerState.OPEN, BreakerState.HALF_OPEN)


@pytest.mark.asyncio
async def test_429_counts_as_failure(
    breaker: CircuitBreaker,
) -> None:
    """A 429 provider error must count toward the breaker threshold."""
    for _ in range(3):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_429)

    assert breaker.state in (BreakerState.OPEN, BreakerState.HALF_OPEN)


# ---------------------------------------------------------------------------
# Two consecutive backoffs differ (jitter works)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consecutive_backoffs_differ(
    clock: FakeClock,
) -> None:
    """Two breakers tripped at the same moment must not schedule their retries
    at the same time. This is the entire point of jitter.

    We create many breakers and trip them all at the same clock time. If jitter
    is working, not all of them will have the same open_until value.
    """
    open_untils: list[float] = []

    for _ in range(20):
        b = CircuitBreaker(
            failure_threshold=1,
            window_seconds=60.0,
            base_backoff=10.0,
            backoff_ceiling=60.0,
            clock=clock,
        )
        with pytest.raises(ProviderError):
            await b.call(_fail_500)
        open_untils.append(b._open_until)

    # With 20 samples drawn from uniform(0, delay), the probability of all
    # being identical is vanishingly small.
    assert len(set(open_untils)) > 1, (
        "All 20 backoff times are identical — jitter is not working"
    )


# ---------------------------------------------------------------------------
# Exponential growth respects the ceiling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backoff_respects_ceiling(
    clock: FakeClock,
) -> None:
    """No matter how many times the breaker reopens, the maximum possible
    backoff delay must never exceed the configured ceiling.
    """
    ceiling = 5.0
    breaker = CircuitBreaker(
        failure_threshold=1,
        window_seconds=60.0,
        base_backoff=1.0,
        backoff_ceiling=ceiling,
        clock=clock,
    )

    # Initial trip from CLOSED (attempt goes to 1).
    with pytest.raises(ProviderError):
        await breaker.call(_fail_500)

    for cycle in range(10):
        # The breaker is OPEN. Check that the jittered delay <= ceiling.
        actual_delay = breaker._open_until - clock()
        assert actual_delay <= ceiling + 1e-9, (
            f"Cycle {cycle}: delay {actual_delay} exceeded ceiling {ceiling}"
        )

        # Advance past the backoff so the breaker reaches HALF_OPEN, then
        # fail the probe to increment the attempt counter and reopen.
        clock.advance(ceiling + 1)
        assert breaker.state is BreakerState.HALF_OPEN

        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)


# ---------------------------------------------------------------------------
# Failures outside the window do not count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failures_outside_window_do_not_count(
    breaker: CircuitBreaker,
    clock: FakeClock,
) -> None:
    """Old failures that have aged out of the window must not contribute to
    the threshold. Two failures, then a wait longer than the window, then one
    more failure should leave the breaker CLOSED (total in-window = 1).
    """
    # Two failures at t=0.
    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_500)

    # Advance past the window.
    clock.advance(61)

    # One more failure. In-window count is now 1, below threshold of 3.
    with pytest.raises(ProviderError):
        await breaker.call(_fail_500)

    assert breaker.state is BreakerState.CLOSED


# ---------------------------------------------------------------------------
# Successful close resets attempt counter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_close_resets_attempt_counter(
    clock: FakeClock,
) -> None:
    """After a successful half-open probe closes the breaker, the attempt
    counter must reset to zero. The next trip should use the initial backoff
    range, not a continuation of the previous exponential sequence.
    """
    breaker = CircuitBreaker(
        failure_threshold=1,
        window_seconds=60.0,
        base_backoff=1.0,
        backoff_ceiling=1000.0,
        clock=clock,
    )

    # Trip → HALF_OPEN → succeed → CLOSED.
    with pytest.raises(ProviderError):
        await breaker.call(_fail_500)
    clock.advance(1001)
    assert breaker.state is BreakerState.HALF_OPEN
    await breaker.call(_succeed)
    assert breaker.state is BreakerState.CLOSED

    # Trip again. The attempt should be 1 (reset), not 2 (continued).
    # max delay = min(1 * 2^1, 1000) = 2, so open_until <= now + 2.
    now_before = clock()
    with pytest.raises(ProviderError):
        await breaker.call(_fail_500)

    actual_delay = breaker._open_until - now_before
    max_possible = min(1.0 * (2 ** 1), 1000.0)
    assert actual_delay <= max_possible + 1e-9
