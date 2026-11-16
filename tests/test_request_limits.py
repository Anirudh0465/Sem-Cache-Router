# Request size limit tests.
#
# Two layers. The route level tests go through the real app, to prove the
# refusal uses the documented envelope and that nothing reaches a provider. The
# middleware tests drive it directly with hand built ASGI messages, because a
# chunked body and a lying Content-Length cannot be produced reliably through a
# test client, and those are exactly the cases a size limit exists for.

from __future__ import annotations

import json
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.providers.stub import StubProvider
from app.request_limits import BodySizeLimit

AUTH = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
LIMIT = 300


def request_of_size(size: int) -> bytes:
    """A valid chat completion request body that is exactly `size` bytes long."""
    skeleton = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": ""}]})
    padding = size - len(skeleton.encode())
    assert padding >= 0, "size is smaller than the smallest valid request"
    return json.dumps(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x" * padding}]}
    ).encode()


@pytest.fixture
def limited(client: TestClient, settings: Settings) -> TestClient:
    """The shared client, with the body limit lowered so tests stay small."""
    client.app.state.settings = settings.model_copy(update={"max_request_bytes": LIMIT})  # type: ignore[attr-defined]
    return client


def send(client: TestClient, data: bytes, headers: dict[str, str] | None = None) -> Any:
    return client.post("/v1/chat/completions", content=data, headers=headers or AUTH)


# --- through the real app -----------------------------------------------------


def test_a_body_within_the_limit_is_served(limited: TestClient) -> None:
    assert send(limited, request_of_size(LIMIT - 50)).status_code == 200


def test_a_body_exactly_at_the_limit_is_served(limited: TestClient) -> None:
    """The cap is inclusive: `max_request_bytes` bytes are allowed."""
    assert send(limited, request_of_size(LIMIT)).status_code == 200


def test_one_byte_over_the_limit_is_refused(limited: TestClient) -> None:
    assert send(limited, request_of_size(LIMIT + 1)).status_code == 413


def test_the_refusal_uses_the_documented_error_envelope(limited: TestClient) -> None:
    response = send(limited, request_of_size(LIMIT * 4))

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "REQUEST_TOO_LARGE"
    assert str(LIMIT) in error["message"]
    assert error["retry_after_seconds"] is None
    assert error["request_id"].startswith("req_")
    assert response.headers["x-request-id"] == error["request_id"]


def test_an_oversized_request_never_reaches_a_provider(
    limited: TestClient, stub_provider: StubProvider
) -> None:
    send(limited, request_of_size(LIMIT * 4))
    assert stub_provider.call_count == 0


def test_an_inbound_request_id_survives_the_refusal(limited: TestClient) -> None:
    response = send(limited, request_of_size(LIMIT * 4), {**AUTH, "x-request-id": "upstream-42"})
    assert response.json()["error"]["request_id"] == "upstream-42"


def test_a_malformed_body_under_the_limit_is_still_a_400(limited: TestClient) -> None:
    """The limit must not change how an ordinary bad request is reported."""
    response = send(limited, b'{"model": ')
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_endpoints_without_a_body_are_unaffected(limited: TestClient) -> None:
    assert limited.get("/health").status_code == 200
    assert limited.get("/metrics").status_code == 200


# --- settings -------------------------------------------------------------------


def test_the_default_limit_is_one_mebibyte() -> None:
    assert Settings(_env_file=None).max_request_bytes == 1_048_576  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [0, -1])
def test_a_limit_that_admits_nothing_is_rejected_at_startup(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_request_bytes=value)  # type: ignore[call-arg]


# --- the middleware, driven directly -----------------------------------------------


class Channel:
    """The receive side of a connection, fed from a fixed list of messages.

    Counts how many messages were consumed, so a test can assert that reading
    stopped at the chunk that crossed the limit instead of draining the stream.
    """

    def __init__(self, messages: Iterable[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.consumed = 0

    async def __call__(self) -> dict[str, Any]:
        if self.consumed < len(self._messages):
            message = self._messages[self.consumed]
            self.consumed += 1
            return message
        return {"type": "http.disconnect"}


def chunks(*parts: bytes) -> list[dict[str, Any]]:
    """Request messages for a body delivered in the given pieces."""
    return [
        {"type": "http.request", "body": part, "more_body": index < len(parts) - 1}
        for index, part in enumerate(parts)
    ]


def scope_of(
    headers: Iterable[tuple[bytes, bytes]] = (), state: Any = None, kind: str = "http"
) -> dict[str, Any]:
    """An ASGI scope whose app carries the given settings, or none."""
    return {
        "type": kind,
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": list(headers),
        "app": SimpleNamespace(state=state if state is not None else with_limit(LIMIT)),
    }


def with_limit(limit: int) -> SimpleNamespace:
    return SimpleNamespace(settings=SimpleNamespace(max_request_bytes=limit))


class Downstream:
    """A stand in for the app behind the middleware, recording what it was given."""

    def __init__(self) -> None:
        self.called = False
        self.body = b""
        self.after_body: dict[str, Any] | None = None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.called = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.after_body = await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


async def run(
    scope: dict[str, Any], channel: Channel
) -> tuple[Downstream, list[dict[str, Any]]]:
    downstream = Downstream()
    sent: list[dict[str, Any]] = []

    async def send_message(message: dict[str, Any]) -> None:
        sent.append(message)

    await BodySizeLimit(downstream)(scope, channel, send_message)
    return downstream, sent


def status_of(sent: list[dict[str, Any]]) -> int:
    return int(sent[0]["status"])


async def test_a_chunked_body_over_the_limit_is_refused() -> None:
    """No Content-Length at all, so only counting the bytes can catch it."""
    channel = Channel(chunks(b"a" * 200, b"b" * 200, b"c" * 200))
    downstream, sent = await run(scope_of(), channel)

    assert status_of(sent) == 413
    assert not downstream.called


async def test_reading_stops_at_the_chunk_that_crosses_the_limit() -> None:
    """An oversized stream costs one chunk past the cap, not the whole stream."""
    channel = Channel(chunks(b"a" * 200, b"b" * 200, b"c" * 200, b"d" * 200))
    await run(scope_of(), channel)

    assert channel.consumed == 2


async def test_a_declared_length_over_the_limit_is_refused_without_reading() -> None:
    channel = Channel(chunks(b"a" * 10))
    downstream, sent = await run(scope_of([(b"content-length", b"5000")]), channel)

    assert status_of(sent) == 413
    assert not downstream.called
    assert channel.consumed == 0


async def test_a_lying_content_length_does_not_get_a_large_body_through() -> None:
    """Declares 10 bytes, sends 600. The declared length cannot be what is trusted."""
    channel = Channel(chunks(b"a" * 300, b"b" * 300))
    downstream, sent = await run(scope_of([(b"content-length", b"10")]), channel)

    assert status_of(sent) == 413
    assert not downstream.called


async def test_a_malformed_content_length_is_ignored_and_the_bytes_decide() -> None:
    small, sent_small = await run(
        scope_of([(b"content-length", b"abc")]), Channel(chunks(b"a" * 50))
    )
    large, sent_large = await run(
        scope_of([(b"content-length", b"abc")]), Channel(chunks(b"a" * 400))
    )

    assert status_of(sent_small) == 200 and small.called
    assert status_of(sent_large) == 413 and not large.called


async def test_a_body_within_the_limit_reaches_the_app_unchanged() -> None:
    """Delivered in pieces, replayed as one message holding exactly the same bytes."""
    channel = Channel(chunks(b"first-", b"second-", b"third"))
    downstream, sent = await run(scope_of(), channel)

    assert status_of(sent) == 200
    assert downstream.body == b"first-second-third"


async def test_the_connection_state_after_the_body_still_comes_from_the_server() -> None:
    """The disconnect that follows the body is the real one, not something invented."""
    downstream, _ = await run(scope_of(), Channel(chunks(b"hello")))
    assert downstream.after_body == {"type": "http.disconnect"}


async def test_a_body_exactly_at_the_limit_is_allowed() -> None:
    downstream, sent = await run(scope_of(), Channel(chunks(b"a" * LIMIT)))
    assert status_of(sent) == 200 and downstream.called


async def test_a_client_that_disconnects_mid_body_is_not_reported_as_too_large() -> None:
    channel = Channel([{"type": "http.request", "body": b"partial", "more_body": True}])
    downstream, sent = await run(scope_of(), channel)

    assert status_of(sent) == 200
    assert downstream.body == b"partial"


async def test_a_non_http_scope_passes_straight_through() -> None:
    """The lifespan scope must reach the app untouched, with no body handling."""
    seen: list[Any] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append((scope["type"], receive))

    channel = Channel([])
    await BodySizeLimit(app)(scope_of(kind="lifespan"), channel, lambda message: None)  # type: ignore[arg-type,return-value]

    assert seen == [("lifespan", channel)]
    assert channel.consumed == 0


async def test_the_environment_supplies_the_limit_before_the_lifespan_has_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no settings on app.state yet, the limit comes from the environment."""
    monkeypatch.setenv("SEMCACHE_MAX_REQUEST_BYTES", "50")

    _, refused = await run(scope_of(state=SimpleNamespace()), Channel(chunks(b"a" * 51)))
    downstream, served = await run(scope_of(state=SimpleNamespace()), Channel(chunks(b"a" * 50)))

    assert status_of(refused) == 413
    assert status_of(served) == 200 and downstream.called
