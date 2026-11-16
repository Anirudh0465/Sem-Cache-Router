# Provider and failover wiring tests.
#
# Three layers. The factory is built and inspected on its own. The real adapters,
# router and breakers are then run against a fake HTTP upstream, which is failover
# as production would run it, minus the vendors. And the route is driven through
# a seeded router, to prove what the caller, the metrics and the log line see.
#
# The fake upstream is httpx's mock transport, so no test opens a socket and none
# needs a key.

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.models import ChatCompletionRequest, Message
from app.providers import factory
from app.providers.base import ProviderError
from app.providers.breaker import CircuitBreaker
from app.providers.factory import (
    DEFAULT_FALLBACK_MODEL,
    build_router,
    install_providers,
    uses_a_real_provider,
)
from app.providers.openai_client import OpenAIClient
from app.providers.router import AllProvidersFailed, FailoverRouter, ProviderSlot
from app.providers.stub import StubProvider
from tests.conftest import router_around
from tests.test_failover_router import ScriptedProvider

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"
OPENAI = "http://openai.test/v1"
ANTHROPIC = "http://anthropic.test/v1"

OPENAI_REPLY = {
    "choices": [
        {"message": {"role": "assistant", "content": "from openai"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
}
ANTHROPIC_REPLY = {
    "content": [{"type": "text", "text": "from anthropic"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def settings_with(**overrides: Any) -> Settings:
    """Both vendors configured against fake endpoints, ignoring any local .env."""
    values: dict[str, Any] = {
        "primary_provider": "openai",
        "secondary_provider": "anthropic",
        "OPENAI_API_KEY": "openai-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
        "openai_base_url": OPENAI,
        "anthropic_base_url": ANTHROPIC,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def chat(model: str = MODEL) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


class Upstream:
    """A fake pair of vendors that records every request and answers as told."""

    def __init__(self) -> None:
        self.openai_status = 200
        self.anthropic_status = 200
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if request.url.host == "openai.test":
            if self.openai_status != 200:
                return httpx.Response(self.openai_status, json={"error": "upstream"})
            return httpx.Response(200, json=OPENAI_REPLY)
        if self.anthropic_status != 200:
            return httpx.Response(self.anthropic_status, json={"error": "upstream"})
        return httpx.Response(200, json=ANTHROPIC_REPLY)

    def calls_to(self, host: str) -> list[httpx.Request]:
        return [request for request in self.seen if request.url.host == host]

    @property
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


def real_router(settings: Settings, upstream: Upstream) -> FailoverRouter:
    client = upstream.client
    primary = OpenAIClient(client, settings)
    return build_router(settings, client, primary)


# --- building from settings -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "secondary", "expected"),
    [
        ("stub", None, False),
        ("openai", None, True),
        ("stub", "anthropic", True),
        ("openai", "anthropic", True),
    ],
)
def test_a_real_provider_is_recognised_in_either_role(
    primary: str, secondary: str | None, expected: bool
) -> None:
    configured = Settings(_env_file=None, primary_provider=primary, secondary_provider=secondary)  # type: ignore[call-arg]
    assert uses_a_real_provider(configured) is expected


async def test_the_default_setup_builds_a_stub_behind_a_breaker_and_opens_no_pool() -> None:
    state = type("State", (), {})()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    client = await install_providers(state, settings)

    assert client is None
    assert isinstance(state.provider, StubProvider)
    assert state.router.primary is state.provider
    assert state.router.breaker_states() == {"stub": "closed"}


async def test_a_seeded_provider_is_kept_and_the_router_is_built_around_it() -> None:
    seeded = StubProvider()
    state = type("State", (), {})()
    state.provider = seeded

    await install_providers(state, Settings(_env_file=None))  # type: ignore[call-arg]

    assert state.provider is seeded
    assert state.router.primary is seeded


async def test_a_seeded_router_is_never_replaced() -> None:
    seeded = router_around(StubProvider())
    state = type("State", (), {})()
    state.provider = StubProvider()
    state.router = seeded

    client = await install_providers(state, settings_with())

    assert state.router is seeded
    assert client is None


async def test_both_providers_are_built_and_each_gets_its_own_breaker(upstream: Upstream) -> None:
    router = real_router(settings_with(), upstream)
    assert router.breaker_states() == {"openai": "closed", "anthropic": "closed"}


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, DEFAULT_FALLBACK_MODEL["anthropic"]),
        ("claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20240620"),
    ],
)
async def test_the_fallback_model_is_the_configured_one_or_the_providers_default(
    upstream: Upstream, configured: str | None, expected: str
) -> None:
    upstream.openai_status = 503
    router = real_router(settings_with(secondary_model=configured), upstream)

    await router.complete(chat("gpt-4o-mini"))

    sent = json.loads(upstream.calls_to("anthropic.test")[0].content)
    assert sent["model"] == expected


async def test_a_missing_secondary_key_names_the_setting_to_change() -> None:
    state = type("State", (), {})()
    settings = settings_with(ANTHROPIC_API_KEY=None)

    with pytest.raises(ValueError, match="SEMCACHE_SECONDARY_PROVIDER is anthropic"):
        await install_providers(state, settings)


async def test_a_startup_failure_does_not_leave_a_connection_pool_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[httpx.AsyncClient] = []

    class Tracked(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(factory.httpx, "AsyncClient", Tracked)
    state = type("State", (), {})()

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await install_providers(state, settings_with(OPENAI_API_KEY=None))

    assert len(opened) == 1 and opened[0].is_closed


# --- the real adapters, router and breakers against a fake upstream -----------------------------


async def test_a_healthy_primary_answers_and_the_secondary_is_never_contacted(
    upstream: Upstream,
) -> None:
    completion = await real_router(settings_with(), upstream).complete(chat())

    assert completion.provider == "openai"
    assert completion.response.message.content == "from openai"
    assert upstream.calls_to("anthropic.test") == []


async def test_a_failing_openai_is_answered_by_anthropic_with_its_own_model_and_price(
    upstream: Upstream,
) -> None:
    """TEST-013 with the real adapters: the caller gets an answer, priced as Anthropic's."""
    upstream.openai_status = 503

    completion = await real_router(settings_with(), upstream).complete(chat())

    assert completion.provider == "anthropic"
    assert completion.failed_over is True
    assert completion.response.message.content == "from anthropic"
    expected_cost = 100 * 0.25e-6 + 50 * 1.25e-6
    assert completion.response.cost_usd == pytest.approx(expected_cost)


async def test_the_breaker_threshold_comes_from_settings_and_stops_calls_to_the_primary(
    upstream: Upstream,
) -> None:
    upstream.openai_status = 503
    router = real_router(settings_with(breaker_failure_threshold=2), upstream)

    for _ in range(2):
        await router.complete(chat())
    assert router.breaker_states() == {"openai": "open", "anthropic": "closed"}
    contacted = len(upstream.calls_to("openai.test"))

    await router.complete(chat())

    assert len(upstream.calls_to("openai.test")) == contacted


async def test_a_request_openai_rejects_as_malformed_is_not_sent_to_anthropic(
    upstream: Upstream,
) -> None:
    upstream.openai_status = 400
    router = real_router(settings_with(), upstream)

    with pytest.raises(ProviderError) as raised:
        await router.complete(chat())

    assert raised.value.status_code == 400
    assert upstream.calls_to("anthropic.test") == []


async def test_both_vendors_down_fails_with_every_attempt_named(upstream: Upstream) -> None:
    upstream.openai_status = 503
    upstream.anthropic_status = 500

    with pytest.raises(AllProvidersFailed) as raised:
        await real_router(settings_with(), upstream).complete(chat())

    assert [a.provider for a in raised.value.attempts] == ["openai", "anthropic"]


# --- through the route ---------------------------------------------------------------------------


@pytest.fixture
def routed(client: TestClient) -> tuple[TestClient, ScriptedProvider, ScriptedProvider]:
    """The shared client with a two provider router seeded in place of its default."""
    primary, secondary = ScriptedProvider("primary"), ScriptedProvider("secondary")
    client.app.state.router = FailoverRouter(  # type: ignore[attr-defined]
        [
            ProviderSlot(primary, CircuitBreaker(failure_threshold=3)),
            ProviderSlot(secondary, CircuitBreaker(failure_threshold=3), "fallback"),
        ]
    )
    return client, primary, secondary


def ask(client: TestClient, content: str = "a question never asked before") -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
        headers=AUTH,
    )


def test_a_failing_primary_is_invisible_to_the_caller(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, secondary = routed
    primary.fail_with(503)

    response = ask(client)

    assert response.status_code == 200
    assert response.json()["semcache_status"] == "miss"
    assert response.json()["choices"][0]["message"]["content"] == "answered by secondary"


def test_the_failover_is_counted_once_per_request_that_needed_it(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, _ = routed
    ask(client, "healthy one")
    assert client.get("/metrics").json()["breaker_failovers"] == 0

    primary.fail_with(503)
    ask(client, "needs failover one")
    ask(client, "needs failover two")

    assert client.get("/metrics").json()["breaker_failovers"] == 2


def test_an_answer_reached_by_failover_is_cached_like_any_other(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, secondary = routed
    primary.fail_with(503)
    ask(client, "same question twice")

    again = ask(client, "same question twice")

    assert again.json()["semcache_status"] == "exact_hit"
    assert secondary.calls == 1


def test_the_log_line_names_the_provider_that_answered_not_the_one_asked_first(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, primary, _ = routed
    primary.fail_with(503)

    with caplog.at_level("INFO", logger="semcache.request"):
        ask(client)

    line = json.loads(next(r for r in caplog.records if r.name == "semcache.request").getMessage())
    assert line["provider"] == "secondary"
    assert line["breaker_state"] == "closed"


def test_every_provider_failing_is_a_502_that_names_no_vendor(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, secondary = routed
    primary.fail_with(503)
    secondary.fail_with(500)

    response = ask(client)

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "ALL_PROVIDERS_UNAVAILABLE"
    assert "primary" not in response.text and "secondary" not in response.text
    assert client.get("/metrics").json()["breaker_failovers"] == 0


def test_a_request_the_provider_rejects_is_a_400_and_is_not_failed_over(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, secondary = routed
    primary.fail_with(400)

    response = ask(client)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert secondary.calls == 0
    assert client.get("/metrics").json()["breaker_failovers"] == 0


def test_a_cache_hit_never_reaches_any_provider(
    routed: tuple[TestClient, ScriptedProvider, ScriptedProvider],
) -> None:
    client, primary, secondary = routed
    ask(client, "warm the cache")
    calls = primary.calls

    ask(client, "warm the cache")

    assert primary.calls == calls and secondary.calls == 0


# --- the lifespan -------------------------------------------------------------------------------


def app_without_providers(settings: Settings, fake_redis: aioredis.FakeRedis) -> Any:
    app = main.create_app()
    app.state.settings = settings
    app.state.redis = fake_redis
    return app


def test_the_lifespan_builds_the_router_from_settings_and_closes_the_pool_it_opened(
    fake_redis: aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[httpx.AsyncClient] = []

    class Tracked(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(factory.httpx, "AsyncClient", Tracked)
    app = app_without_providers(settings_with(), fake_redis)

    with TestClient(app):
        assert app.state.router.breaker_states() == {"openai": "closed", "anthropic": "closed"}
        assert app.state.provider.name == "openai"
        assert not opened[0].is_closed

    assert opened[0].is_closed


def test_the_default_lifespan_opens_no_pool_and_serves_from_the_stub(
    fake_redis: aioredis.FakeRedis,
) -> None:
    app = app_without_providers(Settings(_env_file=None), fake_redis)  # type: ignore[call-arg]

    with TestClient(app) as client:
        assert app.state.router.breaker_states() == {"stub": "closed"}
        assert ask(client).status_code == 200


def test_a_gateway_asked_for_failover_without_the_key_refuses_to_start(
    fake_redis: aioredis.FakeRedis,
) -> None:
    app = app_without_providers(settings_with(ANTHROPIC_API_KEY=None), fake_redis)

    with pytest.raises(ValueError, match="SEMCACHE_SECONDARY_PROVIDER"), TestClient(app):
        pass


def test_the_primary_breaker_state_reaches_the_span_and_log_when_nothing_is_wired(
    client: TestClient,
) -> None:
    """With no router at all the state is reported as absent, not as a closed breaker."""
    from app.telemetry.request_record import NO_BREAKER, breaker_state_of

    assert breaker_state_of(type("State", (), {})()) == NO_BREAKER
    assert breaker_state_of(client.app.state) == "closed"  # type: ignore[attr-defined]
