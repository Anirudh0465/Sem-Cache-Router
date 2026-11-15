# Trace detail tests: child spans and trace_id log correlation.
#
# A gateway miss produces four spans: a root span ("chat <model>") and three
# child spans — "cache tier1 lookup", "cache tier2 lookup", and
# "provider call <model>". A Tier 1 hit skips both Tier 2 and the provider,
# so it produces only two spans.
#
# trace_id in the log line ties a log grep to its Jaeger trace. When tracing
# is off (no tracer_provider on state) the span is a no-op whose context is
# invalid, so trace_id is null — an honest null beats a fake all-zero string.

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"


def body(content: str = "What is the capital of France?") -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": content}]}


@pytest.fixture
def exporter(client: TestClient) -> Iterator[InMemorySpanExporter]:
    """In-memory exporter wired to the app's tracer provider."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    client.app.state.tracer_provider = provider  # type: ignore[attr-defined]
    yield exp
    provider.shutdown()


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in exporter.get_finished_spans()}


def _log_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [
        json.loads(r.getMessage())
        for r in caplog.records
        if r.name == "semcache.request"
    ]


# --- child spans on a miss ---------------------------------------------------


def test_a_miss_produces_tier1_child_span(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Tier 1 lookup is instrumented on every request."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert "cache tier1 lookup" in _spans_by_name(exporter)


def test_a_miss_produces_tier2_child_span(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Tier 2 lookup is instrumented when tier 1 misses."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert "cache tier2 lookup" in _spans_by_name(exporter)


def test_a_miss_produces_provider_child_span(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Provider call is instrumented on a full cache miss."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert f"provider call {MODEL}" in _spans_by_name(exporter)


def test_a_miss_produces_exactly_four_spans(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Root + tier1 + tier2 + provider = four spans on a full miss."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert len(spans) == 4, names


# --- child spans on a Tier 1 hit ---------------------------------------------


def test_a_tier1_hit_produces_only_two_spans(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """A Tier 1 hit returns before reaching Tier 2 or the provider."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    exporter.clear()
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert len(spans) == 2, names
    assert "cache tier1 lookup" in names
    assert f"chat {MODEL}" in names
    assert "provider call gpt-4o-mini" not in names


# --- child span parentage ----------------------------------------------------


def test_child_spans_are_children_of_root(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """Every child span's parent is the root span, not another child."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.parent is None)
    children = [s for s in spans if s.parent is not None]
    root_span_id = root.context.span_id  # type: ignore[union-attr]
    for child in children:
        assert child.parent.span_id == root_span_id, (  # type: ignore[union-attr]
            f"{child.name!r} has wrong parent"
        )


# --- trace_id in log ---------------------------------------------------------


def test_trace_id_is_null_when_tracing_is_off(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """With no tracer_provider, the span is a no-op; trace_id must be null."""
    client.app.state.tracer_provider = None  # type: ignore[attr-defined]
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)
    lines = _log_lines(caplog)
    assert len(lines) == 1
    assert lines[0]["trace_id"] is None


def test_trace_id_is_32_hex_chars_when_tracing_is_on(
    client: TestClient,
    exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With a real tracer, trace_id is a 32-character lowercase hex string."""
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)
    lines = _log_lines(caplog)
    assert len(lines) == 1
    tid = lines[0]["trace_id"]
    assert tid is not None
    assert re.fullmatch(r"[0-9a-f]{32}", tid), f"unexpected trace_id: {tid!r}"


def test_log_trace_id_matches_root_span_trace_id(
    client: TestClient,
    exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The trace_id in the log line matches the root span's trace ID."""
    with caplog.at_level("INFO", logger="semcache.request"):
        client.post("/v1/chat/completions", json=body(), headers=AUTH)

    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.parent is None)
    expected_tid = format(root.context.trace_id, "032x")  # type: ignore[union-attr]

    lines = _log_lines(caplog)
    assert lines[0]["trace_id"] == expected_tid


# --- provider fail span -------------------------------------------------------


def test_failed_request_has_no_provider_child_span(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """A missing API key fails before the provider is called — no provider span."""
    client.post("/v1/chat/completions", json=body())  # no AUTH header
    names = [s.name for s in exporter.get_finished_spans()]
    assert f"provider call {MODEL}" not in names


def test_stub_provider_name_appears_in_provider_span(
    client: TestClient, exporter: InMemorySpanExporter
) -> None:
    """The provider child span name reflects the model that was actually called."""
    client.post("/v1/chat/completions", json=body(), headers=AUTH)
    spans = _spans_by_name(exporter)
    assert f"provider call {MODEL}" in spans
