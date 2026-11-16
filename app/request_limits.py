# A cap on how large a request body may be.
#
# PRD section 26 says request bodies are size limited. Without a cap, a caller
# can make the gateway read and parse an arbitrarily large JSON document before
# any check runs, and the same text would then be embedded and sent to a
# provider that bills by the token.
#
# A plain ASGI middleware rather than Starlette's BaseHTTPMiddleware, and one
# that buffers rather than raising from inside the stream. FastAPI turns any
# exception raised while it reads the body into a generic 400, so rejecting from
# inside receive() would report a size problem as a malformed request. Buffering
# up to the cap lets the rejection happen here, with the documented envelope,
# and the route then sees the body replayed exactly as it arrived. Memory is
# bounded by the cap, which is small, because a chat request is a short JSON
# document and there is nothing to gain from streaming one.

from __future__ import annotations

from fastapi import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import get_settings
from app.errors import error_response, request_id
from app.models import ErrorCode


class BodySizeLimit:
    """Reject, with a 413, any request whose body exceeds max_request_bytes.

    The limit is read from app.state.settings on every request, not captured at
    construction, so it follows the same test seam as every other dependency: a
    test replaces the settings on app.state and the next request sees them.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = _limit_for(scope)

        # A declared length over the cap is refused without reading a byte of the
        # body, which is the common case for an honest client sending too much.
        if _declared_length(scope) > limit:
            await _reject(scope, receive, send, limit)
            return

        # A declared length cannot be trusted, and a chunked body has none, so the
        # bytes actually received are counted as well.
        buffered = await _read_up_to(receive, limit)
        if buffered is None:
            await _reject(scope, receive, send, limit)
            return

        await self.app(scope, _replay(buffered, receive), send)


def _limit_for(scope: Scope) -> int:
    """The configured cap, from app.state if the lifespan has set it, else the environment."""
    settings = getattr(scope["app"].state, "settings", None) or get_settings()
    limit: int = settings.max_request_bytes
    return limit


def _declared_length(scope: Scope) -> int:
    """The Content-Length the client claimed, or 0 when absent or not a number.

    A malformed value is not trusted and not an error here: the byte count that
    follows is what actually enforces the cap.
    """
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


async def _read_up_to(receive: Receive, limit: int) -> bytes | None:
    """Collect the whole body, or return None as soon as it passes the limit.

    Stops reading at the first chunk that crosses the cap, so an oversized
    stream costs at most one chunk beyond it. A client that disconnects first
    leaves an empty body, which the route then rejects as invalid on its own.
    """
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] != "http.request":
            return bytes(body)
        body += message.get("body", b"")
        if len(body) > limit:
            return None
        if not message.get("more_body", False):
            return bytes(body)


def _replay(body: bytes, receive: Receive) -> Receive:
    """A receive callable that hands the buffered body over once, then defers.

    The first call returns the whole body as a single message. Anything after
    that, such as the disconnect the server sends when the client goes away,
    still comes from the real channel, so the app's view of the connection is
    unchanged.
    """
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
    """Answer 413 in the documented error envelope."""
    response = error_response(
        413,
        ErrorCode.REQUEST_TOO_LARGE,
        f"Request body exceeds the limit of {limit} bytes.",
        request_id(Request(scope)),
    )
    await response(scope, receive, send)
