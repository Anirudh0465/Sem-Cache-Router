# Tier 2 degradation visibility tests.
#
# When Chroma is unreachable the gateway must (a) still return 200,
# (b) increment the tier2_unavailable counter once per request, (c) write
# "unavailable" into the structured log line, and (d) set the
# semcache.tier2_status span attribute to "unavailable".
#
# "disabled" and "ok" are also tested so that the full range of tier2_status
# values is covered and a future refactor cannot silently collapse them.

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.main import create_app
from app.providers.stub import StubProvider
from tests.conftest import FakeClock, MappingEmbedder

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"
PROMPT = "What is the capital of France?"
COOLDOWN = 30.0


def _body(content: str = PROMPT) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": content}]}


class AlwaysDownFactory:
    """A Chroma client factory that never succeeds."""

    def __call__(self) -> Any:
        raise ConnectionError("Chroma is down")


def _make_app(
    semantic_cache: Any = None,
    *,
    tracer_provider: TracerProvider | None = None,
) -> tuple[Any, aioredis.FakeRedis]:
    """Build an isolated app with fakes, optionally injecting a semantic cache."""
    app = create_app()
    redis = aioredis.FakeRedis(decode_responses=True)
    app.state.settings = Settings(redis_url="redis://localhost:6379/0", primary_provider="stub")
    app.state.redis = redis
    app.state.provider = StubProvider()
    app.state.cache = ExactCache(redis, 3600)
    app.state.limiter = None
    app.state.semantic_cache = semantic_cache
    if tracer_provider is not None:
        app.state.tracer_provider = tracer_provider
    return app, redis


def _log_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "semcache.request"]


# ---------------------------------------------------------------------------
# Counter: Chroma down
# ---------------------------------------------------------------------------


def test_chroma_down_request_still_returns_200(fake_clock: FakeClock) -> None:
    """TEST-015 at the counter level. The request must succeed despite the outage."""
    sc = SemanticCache(
        AlwaysDownFactory(),
        MappingEmbedder(),
        collection_name="test",
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    app, _ = _make_app(sc)
    with TestClient(app) as client:
        r = client.post("/v1/chat/completions", json=_body(), headers=AUTH)
    assert r.status_code == 200


def test_chroma_down_increments_tier2_unavailable(fake_clock: FakeClock) -> None:
    """One request with Chroma down: tier2_unavailable becomes 1."""
    sc = SemanticCache(
        AlwaysDownFactory(),
        MappingEmbedder(),
        collection_name="test",
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    app, _ = _make_app(sc)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_body(), headers=AUTH)
        metrics = client.get("/metrics").json()
    assert metrics["tier2_unavailable"] == 1


def test_three_requests_during_cooldown_count_three(fake_clock: FakeClock) -> None:
    """Each request inside the reconnect cooldown counts once; no double counting.

    Three distinct prompts are used so each is a Tier 1 miss and must go
    through the Tier 2 step.  If the same prompt were reused, the second and
    third would be Tier 1 exact hits and never reach Tier 2 at all.
    """
    sc = SemanticCache(
        AlwaysDownFactory(),
        MappingEmbedder(),
        collection_name="test",
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    app, _ = _make_app(sc)
    prompts = ["Question alpha", "Question beta", "Question gamma"]
    with TestClient(app) as client:
        for prompt in prompts:
            # Advance time less than the cooldown so we stay within it.
            fake_clock.advance(1.0)
            r = client.post("/v1/chat/completions", json=_body(prompt), headers=AUTH)
            assert r.status_code == 200
        metrics = client.get("/metrics").json()
    assert metrics["tier2_unavailable"] == 3


# ---------------------------------------------------------------------------
# Log line: tier2_status field
# ---------------------------------------------------------------------------


def test_chroma_down_writes_unavailable_to_log(
    caplog: pytest.LogCaptureFixture, fake_clock: FakeClock
) -> None:
    """The structured log line says 'unavailable', not 'miss'."""
    sc = SemanticCache(
        AlwaysDownFactory(),
        MappingEmbedder(),
        collection_name="test",
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    app, _ = _make_app(sc)
    with TestClient(app) as client:
        with caplog.at_level("INFO", logger="semcache.request"):
            client.post("/v1/chat/completions", json=_body(), headers=AUTH)
    lines = _log_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["tier2_status"] == "unavailable"


def test_tier2_disabled_writes_disabled_to_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When no semantic cache is configured the log says 'disabled', not null."""
    app, _ = _make_app(semantic_cache=None)
    with TestClient(app) as client:
        with caplog.at_level("INFO", logger="semcache.request"):
            client.post("/v1/chat/completions", json=_body(), headers=AUTH)
    lines = _log_lines(caplog)
    assert lines[0]["tier2_status"] == "disabled"


def test_genuine_miss_writes_ok_to_log(
    caplog: pytest.LogCaptureFixture,
    chroma_factory: Callable[[], Any],
    collection_name: str,
    fake_clock: FakeClock,
) -> None:
    """When Chroma is reachable the log says 'ok', whether or not it was a hit."""
    sc = SemanticCache(
        chroma_factory,
        MappingEmbedder(),
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    app, _ = _make_app(sc)
    with TestClient(app) as client:
        with caplog.at_level("INFO", logger="semcache.request"):
            client.post("/v1/chat/completions", json=_body(), headers=AUTH)
    lines = _log_lines(caplog)
    assert lines[0]["tier2_status"] == "ok"


# ---------------------------------------------------------------------------
# Span attribute: semcache.tier2_status
# ---------------------------------------------------------------------------


def test_chroma_down_sets_span_attribute_unavailable(fake_clock: FakeClock) -> None:
    """The span carries semcache.tier2_status = unavailable on a Chroma outage."""
    sc = SemanticCache(
        AlwaysDownFactory(),
        MappingEmbedder(),
        collection_name="test",
        ttl_seconds=60,
        threshold=0.90,
        reconnect_seconds=COOLDOWN,
        clock=fake_clock,
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    app, _ = _make_app(sc, tracer_provider=provider)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_body(), headers=AUTH)
    provider.shutdown()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("semcache.tier2_status") == "unavailable"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Disabled does not bump the counter
# ---------------------------------------------------------------------------


def test_tier2_disabled_does_not_increment_unavailable_counter() -> None:
    """'disabled' is not an outage; the counter must not move."""
    app, _ = _make_app(semantic_cache=None)
    with TestClient(app) as client:
        client.post("/v1/chat/completions", json=_body(), headers=AUTH)
        metrics = client.get("/metrics").json()
    assert metrics["tier2_unavailable"] == 0
