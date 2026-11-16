import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from httpx import Response

from app.config import Settings
from app.models import ChatCompletionRequest, Message
from app.providers.anthropic_client import AnthropicClient
from app.providers.base import ProviderError


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: Response):
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> Response:
        self._response.request = request
        return self._response


@pytest.mark.asyncio
async def test_anthropic_client_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMCACHE_ANTHROPIC_API_KEY", "test_key")

    mock_response = httpx.Response(
        200,
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "model": "claude-3-haiku-20240307",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
    )

    client = httpx.AsyncClient(transport=MockTransport(mock_response))
    adapter = AnthropicClient(http_client=client)

    req = ChatCompletionRequest(
        model="claude-3-haiku-20240307",
        messages=[
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi!")
        ]
    )

    res = await adapter.complete(req)
    assert res.provider == "anthropic"
    assert res.message.content == "Hello, world!"
    assert res.usage.prompt_tokens == 10
    assert res.usage.completion_tokens == 5
    assert res.finish_reason == "stop"


@pytest.mark.asyncio
async def test_anthropic_client_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMCACHE_ANTHROPIC_API_KEY", "test_key")

    mock_response = httpx.Response(
        400,
        json={"error": {"type": "invalid_request_error", "message": "Bad request"}},
    )

    client = httpx.AsyncClient(transport=MockTransport(mock_response))
    adapter = AnthropicClient(http_client=client)

    req = ChatCompletionRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="Hi!")]
    )

    with pytest.raises(ProviderError) as exc:
        await adapter.complete(req)

    assert exc.value.status_code == 400
    assert "Anthropic HTTP error" in str(exc.value)


# --- the endpoint, the key, the timeout and how failures arrive ---------------------------

UPSTREAM = "http://upstream.test/v1"

REPLY = {
    "content": [{"type": "text", "text": "Paris."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def settings_with(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"ANTHROPIC_API_KEY": "test-key", "anthropic_base_url": UPSTREAM}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


def adapter_over(
    handler: Callable[[httpx.Request], httpx.Response], **overrides: Any
) -> AnthropicClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicClient(client, settings_with(**overrides))


def ask(*messages: Message) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="claude-3-haiku-20240307", messages=list(messages))


async def test_the_request_goes_to_the_configured_endpoint_with_the_key_and_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    adapter = adapter_over(handler, provider_timeout_seconds=7)
    await adapter.complete(ask(Message(role="user", content="hi")))

    assert str(seen[0].url) == f"{UPSTREAM}/messages"
    assert seen[0].headers["x-api-key"] == "test-key"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert seen[0].extensions["timeout"]["read"] == 7


async def test_system_messages_become_the_system_field_not_a_message() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=REPLY)

    await adapter_over(handler).complete(
        ask(
            Message(role="system", content="Be brief."),
            Message(role="system", content="Be kind."),
            Message(role="user", content="hi"),
        )
    )

    sent = json.loads(seen[0].content)
    assert sent["system"] == "Be brief.\nBe kind."
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 1024


@pytest.mark.parametrize("status", [401, 404, 429, 500, 529])
async def test_an_http_error_carries_its_status_code(status: int) -> None:
    adapter = adapter_over(lambda request: httpx.Response(status, json={"error": "nope"}))

    with pytest.raises(ProviderError) as raised:
        await adapter.complete(ask(Message(role="user", content="hi")))

    assert raised.value.status_code == status
    assert raised.value.provider == "anthropic"


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
async def test_a_call_that_got_no_answer_carries_no_status(failure: httpx.HTTPError) -> None:
    """None is what tells the breaker the call never reached the provider, and so counts."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    with pytest.raises(ProviderError) as raised:
        await adapter_over(handler).complete(ask(Message(role="user", content="hi")))

    assert raised.value.status_code is None


def test_construction_without_a_key_is_refused() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(httpx.AsyncClient(), Settings(_env_file=None))  # type: ignore[call-arg]
