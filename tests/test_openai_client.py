# OpenAI adapter tests.
#
# The adapter is the only code that speaks to the vendor, so what it sends and how
# it reads a failure are what the rest of the gateway depends on. These run it
# against httpx's mock transport, which hands the test the request actually built,
# so the URL, the headers and the timeout are checked as they would go out rather
# than assumed.
#
# The failure mapping is the part the circuit breaker lives on. It classifies by
# the status code a ProviderError carries, with None meaning the call never got an
# answer, so each way a call can fail has to arrive as the right one of those.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.models import ChatCompletionRequest, Message, Usage
from app.providers.base import ProviderError
from app.providers.openai_client import OpenAIClient

UPSTREAM = "http://upstream.test/v1"

REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "Paris."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def settings_with(**overrides: Any) -> Settings:
    """Settings with a key and a fake endpoint, ignoring any local .env file."""
    values: dict[str, Any] = {"OPENAI_API_KEY": "test-key", "openai_base_url": UPSTREAM}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def adapter_over(
    handler: Callable[[httpx.Request], httpx.Response], **overrides: Any
) -> OpenAIClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIClient(client, settings_with(**overrides))


def request_for(
    content: str = "What is the capital of France?", **fields: Any
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o-mini", messages=[Message(role="user", content=content)], **fields
    )


# --- what goes out -----------------------------------------------------------------------


async def test_the_request_goes_to_the_configured_endpoint_with_the_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await adapter_over(handler).complete(request_for())

    assert str(seen[0].url) == f"{UPSTREAM}/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"


async def test_the_body_carries_the_model_and_messages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await adapter_over(handler).complete(request_for("hello", max_tokens=50))

    import json

    sent = json.loads(seen[0].content)
    assert sent["model"] == "gpt-4o-mini"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["max_tokens"] == 50


async def test_the_configured_timeout_is_applied() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await adapter_over(handler, provider_timeout_seconds=7).complete(request_for())

    assert seen[0].extensions["timeout"]["read"] == 7


async def test_explicit_settings_win_over_the_process_wide_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan hands over its settings; a different key in the environment must not leak in."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-environment")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await adapter_over(handler).complete(request_for())

    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_construction_without_a_key_is_refused() -> None:
    """Failing at startup, not on the first request that needs the provider."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient(httpx.AsyncClient(), settings)


# --- what comes back ------------------------------------------------------------------------


async def test_a_reply_is_normalised_with_its_usage_and_cost() -> None:
    adapter = adapter_over(lambda request: httpx.Response(200, json=REPLY))

    reply = await adapter.complete(request_for())

    assert reply.provider == "openai"
    assert reply.model == "gpt-4o-mini"
    assert reply.message.content == "Paris."
    assert reply.usage.total_tokens == 15
    assert reply.finish_reason == "stop"
    assert reply.cost_usd == pytest.approx(10 * 0.150e-6 + 5 * 0.600e-6)


def test_price_of_uses_the_committed_table() -> None:
    adapter = adapter_over(lambda request: httpx.Response(200, json=REPLY))
    usage = Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)

    assert adapter.price_of(usage, "gpt-4o-mini") == pytest.approx(0.00075)


def test_the_adapter_names_itself_for_the_breaker_and_the_logs() -> None:
    assert adapter_over(lambda request: httpx.Response(200, json=REPLY)).name == "openai"


# --- how a failure arrives --------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
async def test_an_http_error_carries_its_status_code(status: int) -> None:
    adapter = adapter_over(lambda request: httpx.Response(status, json={"error": "nope"}))

    with pytest.raises(ProviderError) as raised:
        await adapter.complete(request_for())

    assert raised.value.status_code == status
    assert raised.value.provider == "openai"


async def test_a_refused_connection_carries_no_status() -> None:
    """None is what tells the breaker the call never reached the provider, and so counts."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderError) as raised:
        await adapter_over(handler).complete(request_for())

    assert raised.value.status_code is None


async def test_a_timeout_carries_no_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no reply in time")

    with pytest.raises(ProviderError) as raised:
        await adapter_over(handler).complete(request_for())

    assert raised.value.status_code is None
