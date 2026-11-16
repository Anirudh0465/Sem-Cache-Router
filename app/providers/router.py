# Failover across an ordered list of providers, each behind its own circuit breaker.
#
# The breaker decides whether one provider should be called at all. This decides
# what happens next: when the first provider will not do, the request goes to the
# second, in the same call, and the caller never sees the failure. That is the
# whole of PRD BR-008, and what acceptance demonstration 3 observes.
#
# What fails over, and what does not. A provider that is down, slow, rate limiting
# us, or answering with garbage is a reason to try the next one. A provider that
# says the request itself is malformed (400, 404 for an unknown model, 413, 422) is
# not: the next provider would say the same, and retrying would only spend a call
# to learn it. Those go straight back to the caller. A 401 or 403 is our own key
# being refused, which is our fault and not the caller's, so it does fail over,
# though the breaker, which counts only what the PRD says it counts, does not trip
# on it.
#
# A provider that is called with a different model than the caller asked for is a
# fallback, and its rejection is not the caller's fault: the caller never chose
# that model. So a 400 from it counts as a failure, not as a verdict on the request.

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.models import ChatCompletionRequest, ProviderResponse
from app.providers.base import LLMProvider, ProviderError
from app.providers.breaker import CircuitBreaker, CircuitBreakerOpen

log = logging.getLogger(__name__)

# Statuses that mean the request was at fault, so no other provider would do better.
CALLER_DEFECTS = frozenset({400, 404, 413, 422})


@dataclass(frozen=True)
class ProviderSlot:
    """One provider, the breaker guarding it, and the model to call it with.

    model is None for a provider that is called with whatever the caller asked
    for, and a model name for one that is a fallback and needs its own.
    """

    provider: LLMProvider
    breaker: CircuitBreaker
    model: str | None = None


@dataclass(frozen=True)
class Completion:
    """An answer, and how it was reached."""

    response: ProviderResponse
    provider: str
    # The state of the answering provider's breaker when the call was made, so a
    # half open probe is reported as one rather than as the closed it leaves behind.
    breaker_state: str
    # Whether a provider earlier in the order was tried and failed first.
    failed_over: bool


@dataclass(frozen=True)
class Attempt:
    """One provider that was tried and failed, and why."""

    provider: str
    reason: str


class AllProvidersFailed(Exception):
    """Every provider was tried, or had its breaker open, and none answered."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = tuple(attempts)
        detail = ", ".join(f"{attempt.provider} ({attempt.reason})" for attempt in self.attempts)
        super().__init__(f"all providers failed: {detail}")


class FailoverRouter:
    """Try providers in order until one answers."""

    def __init__(self, slots: Sequence[ProviderSlot]) -> None:
        """Hold the ordered slots, which must be non empty and have distinct names.

        Distinct names because the name is how a breaker, a log line and a metric
        are told apart, and two slots sharing one would make them indistinguishable.
        """
        if not slots:
            raise ValueError("a router needs at least one provider")
        names = [slot.provider.name for slot in slots]
        if len(set(names)) != len(names):
            raise ValueError(f"provider names must be distinct, got {names}")
        self._slots = tuple(slots)

    @property
    def primary(self) -> LLMProvider:
        """The first provider, the one tried before any other."""
        return self._slots[0].provider

    def breaker_states(self) -> dict[str, str]:
        """The state of every provider's breaker, by provider name."""
        return {slot.provider.name: slot.breaker.state.value for slot in self._slots}

    async def complete(self, request: ChatCompletionRequest) -> Completion:
        """Answer the request from the first provider that can.

        Raises ProviderError for a request the provider rejected as malformed,
        which is the caller's to hear about, and AllProvidersFailed when no
        provider could answer.
        """
        attempts: list[Attempt] = []
        for index, slot in enumerate(self._slots):
            name = slot.provider.name
            state = slot.breaker.state.value
            outgoing = request if slot.model is None else request.model_copy(
                update={"model": slot.model}
            )
            try:
                response = await slot.breaker.call(slot.provider.complete, outgoing)
            except CircuitBreakerOpen:
                attempts.append(Attempt(name, "circuit open"))
            except ProviderError as exc:
                if exc.status_code in CALLER_DEFECTS and slot.model is None:
                    raise
                log.warning("provider %s failed with status %s; trying the next", name,
                            exc.status_code)
                attempts.append(Attempt(name, f"status {exc.status_code}"))
            except Exception as exc:
                # A reply that cannot be read, say, is as good as no reply. The
                # breaker already counts an exception it cannot classify as a failure.
                log.warning("provider %s failed unexpectedly; trying the next", name, exc_info=True)
                attempts.append(Attempt(name, type(exc).__name__))
            else:
                return Completion(
                    response=response, provider=name, breaker_state=state, failed_over=index > 0
                )
        raise AllProvidersFailed(attempts)
