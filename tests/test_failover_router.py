# Failover router tests.
#
# TEST-013 and TEST-014 from the PRD, plus the rule for what does and does not
# fail over. Providers are scripted to fail in exactly the ways a real one does,
# and each breaker runs on a clock the test moves, so no test sleeps through a
# backoff.

from __future__ import annotations

import asyncio

import pytest

from app.models import ChatCompletionRequest, Message, ProviderResponse, Usage
from app.providers.base import LLMProvider, ProviderError
from app.providers.breaker import CircuitBreaker
from app.providers.router import (
    AllProvidersFailed,
    FailoverRouter,
    ProviderSlot,
)

THRESHOLD = 3
CEILING = 2.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ScriptedProvider(LLMProvider):
    """A provider that answers, or raises whatever the test has set, and records its calls."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failure: BaseException | None = None
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return ProviderResponse(
            provider=self.name,
            model=request.model,
            message=Message(role="assistant", content=f"answered by {self.name}"),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def estimate_tokens(self, messages: list[Message]) -> int:
        return 1

    def price_of(self, usage: Usage, model: str = "") -> float:
        return 0.0

    def fail_with(self, status: int | None) -> None:
        self.failure = ProviderError("boom", provider=self.name, status_code=status)

    def recover(self) -> None:
        self.failure = None

    @property
    def calls(self) -> int:
        return len(self.requests)


def request(model: str = "gpt-4o-mini") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def slot(provider: ScriptedProvider, clock: FakeClock, model: str | None = None) -> ProviderSlot:
    breaker = CircuitBreaker(
        failure_threshold=THRESHOLD,
        window_seconds=60,
        base_backoff=1.0,
        backoff_ceiling=CEILING,
        clock=clock,
    )
    return ProviderSlot(provider, breaker, model)


@pytest.fixture
def primary() -> ScriptedProvider:
    return ScriptedProvider("primary")


@pytest.fixture
def secondary() -> ScriptedProvider:
    return ScriptedProvider("secondary")


@pytest.fixture
def router(
    primary: ScriptedProvider, secondary: ScriptedProvider, clock: FakeClock
) -> FailoverRouter:
    return FailoverRouter([slot(primary, clock), slot(secondary, clock, model="fallback-model")])


# --- the ordinary case ----------------------------------------------------------------


async def test_a_healthy_primary_answers_and_the_secondary_is_never_called(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    completion = await router.complete(request())

    assert completion.provider == "primary"
    assert completion.failed_over is False
    assert completion.breaker_state == "closed"
    assert completion.response.message.content == "answered by primary"
    assert secondary.calls == 0


# --- TEST-013: the primary fails, the caller does not notice --------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 429, None])
async def test_a_failing_primary_is_answered_by_the_secondary_in_the_same_request(
    router: FailoverRouter, primary: ScriptedProvider, status: int | None
) -> None:
    """5xx, a provider rate limit, and a call that never got an answer at all."""
    primary.fail_with(status)

    completion = await router.complete(request())

    assert completion.provider == "secondary"
    assert completion.failed_over is True
    assert completion.response.message.content == "answered by secondary"


async def test_repeated_failures_open_the_breaker_and_the_primary_stops_being_called(
    router: FailoverRouter, primary: ScriptedProvider
) -> None:
    primary.fail_with(503)
    for _ in range(THRESHOLD):
        await router.complete(request())
    assert router.breaker_states()["primary"] == "open"

    calls_before = primary.calls
    completion = await router.complete(request())

    assert primary.calls == calls_before, "an open breaker must not call the provider"
    assert completion.provider == "secondary"


async def test_an_unreadable_reply_fails_over_like_any_other_failure(
    router: FailoverRouter, primary: ScriptedProvider
) -> None:
    primary.failure = KeyError("choices")

    completion = await router.complete(request())

    assert completion.provider == "secondary"


async def test_a_refused_key_fails_over_but_does_not_trip_the_breaker(
    router: FailoverRouter, primary: ScriptedProvider
) -> None:
    """A 401 is our own key being refused. Requests are still answered; the breaker
    counts only what the PRD says it counts, so it stays closed."""
    primary.fail_with(401)

    for _ in range(THRESHOLD * 2):
        completion = await router.complete(request())
        assert completion.provider == "secondary"

    assert router.breaker_states()["primary"] == "closed"


# --- TEST-014: the primary recovers ---------------------------------------------------


async def test_a_successful_probe_after_the_backoff_returns_traffic_to_the_primary(
    router: FailoverRouter, primary: ScriptedProvider, clock: FakeClock
) -> None:
    primary.fail_with(503)
    for _ in range(THRESHOLD):
        await router.complete(request())
    assert router.breaker_states()["primary"] == "open"

    primary.recover()
    clock.advance(CEILING + 1)
    completion = await router.complete(request())

    assert completion.provider == "primary"
    assert completion.failed_over is False
    assert completion.breaker_state == "half_open", "the probe is reported as one"
    assert router.breaker_states()["primary"] == "closed"


async def test_a_failed_probe_reopens_the_breaker_and_the_secondary_still_answers(
    router: FailoverRouter, primary: ScriptedProvider, clock: FakeClock
) -> None:
    primary.fail_with(503)
    for _ in range(THRESHOLD):
        await router.complete(request())

    clock.advance(CEILING + 1)
    completion = await router.complete(request())

    assert completion.provider == "secondary"
    assert router.breaker_states()["primary"] == "open"


# --- what does not fail over ----------------------------------------------------------


@pytest.mark.parametrize("status", [400, 404, 413, 422])
async def test_a_request_the_provider_rejects_as_malformed_goes_back_to_the_caller(
    router: FailoverRouter,
    primary: ScriptedProvider,
    secondary: ScriptedProvider,
    status: int,
) -> None:
    """The next provider would say the same, so asking it only spends a call."""
    primary.fail_with(status)

    with pytest.raises(ProviderError) as raised:
        await router.complete(request())

    assert raised.value.status_code == status
    assert secondary.calls == 0


async def test_malformed_requests_never_trip_the_breaker(
    router: FailoverRouter, primary: ScriptedProvider
) -> None:
    """One badly behaved caller must not be able to fail everyone else over."""
    primary.fail_with(400)

    for _ in range(THRESHOLD * 3):
        with pytest.raises(ProviderError):
            await router.complete(request())

    assert router.breaker_states()["primary"] == "closed"


async def test_a_rejection_from_a_fallback_is_a_failure_not_the_callers_fault(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    """The caller never chose the fallback's model, so its 400 is our misconfiguration."""
    primary.fail_with(503)
    secondary.fail_with(400)

    with pytest.raises(AllProvidersFailed):
        await router.complete(request())


async def test_cancellation_is_not_swallowed(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    primary.failure = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await router.complete(request())

    assert secondary.calls == 0


# --- the fallback's model -------------------------------------------------------------


async def test_the_fallback_gets_its_own_model_and_the_original_request_is_untouched(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    primary.fail_with(503)
    original = request("gpt-4o-mini")

    await router.complete(original)

    assert primary.requests[0].model == "gpt-4o-mini"
    assert secondary.requests[0].model == "fallback-model"
    assert original.model == "gpt-4o-mini"


async def test_a_provider_without_a_model_override_gets_the_callers_model(
    primary: ScriptedProvider, clock: FakeClock
) -> None:
    router = FailoverRouter([slot(primary, clock)])

    await router.complete(request("some-model"))

    assert primary.requests[0].model == "some-model"


# --- when nothing answers -------------------------------------------------------------


async def test_when_every_provider_fails_the_error_names_each_attempt(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    primary.fail_with(503)
    secondary.fail_with(500)

    with pytest.raises(AllProvidersFailed) as raised:
        await router.complete(request())

    assert [(a.provider, a.reason) for a in raised.value.attempts] == [
        ("primary", "status 503"),
        ("secondary", "status 500"),
    ]
    assert "primary (status 503)" in str(raised.value)


async def test_open_breakers_on_every_provider_fail_without_calling_either(
    router: FailoverRouter, primary: ScriptedProvider, secondary: ScriptedProvider
) -> None:
    primary.fail_with(503)
    secondary.fail_with(503)
    for _ in range(THRESHOLD):
        with pytest.raises(AllProvidersFailed):
            await router.complete(request())
    calls = (primary.calls, secondary.calls)

    with pytest.raises(AllProvidersFailed) as raised:
        await router.complete(request())

    assert (primary.calls, secondary.calls) == calls
    assert [a.reason for a in raised.value.attempts] == ["circuit open", "circuit open"]


async def test_a_lone_provider_that_fails_has_nothing_to_fail_over_to(
    primary: ScriptedProvider, clock: FakeClock
) -> None:
    router = FailoverRouter([slot(primary, clock)])
    primary.fail_with(503)

    with pytest.raises(AllProvidersFailed) as raised:
        await router.complete(request())

    assert [a.provider for a in raised.value.attempts] == ["primary"]


# --- the router itself ----------------------------------------------------------------


def test_a_router_needs_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FailoverRouter([])


def test_provider_names_must_be_distinct(clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="distinct"):
        FailoverRouter(
            [slot(ScriptedProvider("same"), clock), slot(ScriptedProvider("same"), clock)]
        )


def test_the_primary_is_the_first_provider(
    router: FailoverRouter, primary: ScriptedProvider
) -> None:
    assert router.primary is primary


def test_every_breaker_state_is_reported_by_provider_name(router: FailoverRouter) -> None:
    assert router.breaker_states() == {"primary": "closed", "secondary": "closed"}
