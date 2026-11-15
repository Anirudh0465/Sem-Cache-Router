# Failover and circuit breaker visibility tests.
#
# These verify that the breaker state is visible in spans and log lines, and
# that a failover to the secondary provider is recorded in metrics. The tests
# use a real CircuitBreaker so the state machine is exercised, not mocked.
#
# To trip the breaker quickly, tests set failure_threshold=1 and hand the
# breaker a StubProvider that fails. After one failure the breaker opens, and
# the next request either goes to the secondary or returns 502 when none is
# configured.
#
# Key behaviour: the call that TRIPS the breaker has already failed (ProviderError
# propagates up). It is SUBSEQUENT calls, when the breaker is OPEN, that are
# routed to the secondary. That is standard circuit breaker semantics.

from __future__ import annotations

import json
from typing import Any

import pytest
from fakeredis import aioredis
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.cache.exact_cache import ExactCache
from app.config import Settings
from app.main import create_app
from app.providers.breaker import CircuitBreaker
from app.providers.stub import StubProvider

AUTH = {"Authorization": "Bearer test-key"}


def _msg(content: str) -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": content}]}


def _base_settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=2,
        primary_provider="stub",
        breaker_failure_threshold=1,
        breaker_window_seconds=60,
        breaker_base_backoff=1.0,
        breaker_backoff_ceiling=60.0,
    )


def _build_client(
    *,
    primary: StubProvider,
    secondary: StubProvider | None = None,
    breaker: CircuitBreaker | None = None,
    exporter: InMemorySpanExporter | None = None,
) -> TestClient:
    """Build an app with the given provider configuration, returning a TestClient.

    Use as a context manager: ``with _build_client(...) as tc:``.
    """
    settings = _base_settings()
    redis = aioredis.FakeRedis(decode_responses=True)
    app = create_app()
    app.state.settings = settings
    app.state.redis = redis
    app.state.provider = primary
    app.state.secondary_provider = secondary
    app.state.cache = ExactCache(redis, settings.cache_ttl_seconds)
    app.state.limiter = None
    if breaker is not None:
        app.state.breaker = breaker
    if exporter is not None:
        otel_provider = TracerProvider()
        otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
        app.state.tracer_provider = otel_provider
    return TestClient(app)


def _fresh_breaker() -> CircuitBreaker:
    """A breaker that trips after a single failure."""
    return CircuitBreaker(failure_threshold=1, window_seconds=60)


def _log_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "semcache.request"
    ]


# --- span attribute -----------------------------------------------------------


def test_closed_breaker_state_appears_in_span() -> None:
    """The span carries breaker_state=closed when the breaker has not tripped."""
    exporter = InMemorySpanExporter()
    with _build_client(primary=StubProvider(), exporter=exporter) as tc:
        tc.post("/v1/chat/completions", json=_msg("hello"), headers=AUTH)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["semcache.breaker_state"] == "closed"


def test_open_breaker_state_appears_in_span_after_trip() -> None:
    """After the primary trips the breaker, subsequent requests see open state."""
    primary = StubProvider(fail_times=1)
    breaker = _fresh_breaker()
    exporter = InMemorySpanExporter()

    with _build_client(
        primary=primary,
        secondary=StubProvider(),
        breaker=breaker,
        exporter=exporter,
    ) as tc:
        # This request fails (primary fails) and trips the breaker.
        tc.post("/v1/chat/completions", json=_msg("first"), headers=AUTH)
        exporter.clear()
        # Next request: breaker is OPEN — breaker_state_of() captures it at request start.
        tc.post("/v1/chat/completions", json=_msg("second"), headers=AUTH)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["semcache.breaker_state"] == "open"


# --- log field ----------------------------------------------------------------


def test_breaker_state_field_appears_in_log(caplog: pytest.LogCaptureFixture) -> None:
    """Every log line carries breaker_state so a query can filter by it."""
    with _build_client(primary=StubProvider()) as tc:
        with caplog.at_level("INFO", logger="semcache.request"):
            tc.post("/v1/chat/completions", json=_msg("hello"), headers=AUTH)

    lines = _log_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["breaker_state"] == "closed"


# --- failover counter ---------------------------------------------------------


def test_failover_counter_increments_on_open_breaker() -> None:
    """When the breaker is OPEN and the secondary answers, breaker_failovers goes up."""
    primary = StubProvider(fail_times=1)
    secondary = StubProvider()
    breaker = _fresh_breaker()

    with _build_client(primary=primary, secondary=secondary, breaker=breaker) as tc:
        # This call fails on primary and trips the breaker (502 returned).
        tc.post("/v1/chat/completions", json=_msg("trip"), headers=AUTH)
        # Breaker is now OPEN: secondary serves without touching primary.
        r = tc.post("/v1/chat/completions", json=_msg("open1"), headers=AUTH)
        assert r.status_code == 200

        metrics = tc.get("/metrics").json()

    assert metrics["breaker_failovers"] == 1


def test_failover_counter_accumulates_across_multiple_open_requests() -> None:
    """Every request served via failover increments the counter."""
    primary = StubProvider(fail_times=1)
    secondary = StubProvider()
    breaker = _fresh_breaker()

    with _build_client(primary=primary, secondary=secondary, breaker=breaker) as tc:
        # Trip the breaker (this request returns 502).
        tc.post("/v1/chat/completions", json=_msg("trip"), headers=AUTH)
        # Three separate requests while breaker is open.
        tc.post("/v1/chat/completions", json=_msg("open1"), headers=AUTH)
        tc.post("/v1/chat/completions", json=_msg("open2"), headers=AUTH)
        tc.post("/v1/chat/completions", json=_msg("open3"), headers=AUTH)

        metrics = tc.get("/metrics").json()

    assert metrics["breaker_failovers"] == 3


def test_failover_counter_stays_at_zero_when_primary_succeeds() -> None:
    """No failover means the counter never moves from zero."""
    with _build_client(primary=StubProvider()) as tc:
        tc.post("/v1/chat/completions", json=_msg("hello"), headers=AUTH)
        metrics = tc.get("/metrics").json()

    assert metrics["breaker_failovers"] == 0


# --- no secondary -------------------------------------------------------------


def test_open_breaker_with_no_secondary_returns_502() -> None:
    """When the breaker is open and there is no secondary, the request fails 502."""
    primary = StubProvider(fail_times=1)
    breaker = _fresh_breaker()

    with _build_client(primary=primary, secondary=None, breaker=breaker) as tc:
        # Trip the breaker — 502 (primary failed, no secondary).
        r1 = tc.post("/v1/chat/completions", json=_msg("trip"), headers=AUTH)
        assert r1.status_code == 502
        assert r1.json()["error"]["code"] == "ALL_PROVIDERS_UNAVAILABLE"

        # Breaker still open, no secondary — still 502.
        r2 = tc.post("/v1/chat/completions", json=_msg("open"), headers=AUTH)
        assert r2.status_code == 502
        assert r2.json()["error"]["code"] == "ALL_PROVIDERS_UNAVAILABLE"


# --- secondary serves correctly -----------------------------------------------


def test_secondary_response_reaches_the_caller() -> None:
    """When failover fires, the caller receives a valid completion from the secondary."""
    primary = StubProvider(fail_times=99)
    secondary = StubProvider()
    breaker = _fresh_breaker()

    with _build_client(primary=primary, secondary=secondary, breaker=breaker) as tc:
        # Trip the breaker (primary fails; secondary not tried on this trip call).
        tc.post("/v1/chat/completions", json=_msg("trip"), headers=AUTH)
        # Now breaker is OPEN: secondary serves cleanly.
        r = tc.post("/v1/chat/completions", json=_msg("served"), headers=AUTH)

    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content != ""
