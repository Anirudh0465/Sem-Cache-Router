# Pins for what all-MiniLM-L6-v2 actually does with the prompts the PRD discusses.
#
# Every other Tier 2 test uses vectors the test chose, which is what lets it
# assert behaviour at an exact similarity. That also means none of them can notice
# if the real model, its tokenizer or the ONNX runtime changes underneath the
# project. These can: they embed the real prompts and pin the measured scores, so
# an upgrade that moves them fails here, loudly, instead of quietly invalidating
# the finding that PRD section 33.4 reports.
#
# The pins are tolerances, not equalities. A different onnxruntime or CPU can move
# the fourth decimal place, and a test that failed on that would be one people
# learned to ignore. The tolerance is far narrower than the gaps the findings rest
# on, so a change large enough to matter is still caught.
#
# Opt in with SEMCACHE_RUN_MODEL_TESTS=1, or scripts/test-model.sh. The weights are
# a 79 MB download on first use, and the default suite promises to need no network.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.cache.embedder import SUPPORTED_MODEL, OnnxEmbedder
from app.cache.semantic_cache import SemanticCache
from app.providers.stub import StubProvider
from evaluation.adversarial import cosine

pytestmark = pytest.mark.model

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"
TOLERANCE = 0.02

# The thresholds the benchmark sweeps, PRD section 21.3.
SWEPT_THRESHOLDS = (0.75, 0.90, 0.95)

# Each pair is the original prompt, the rewrite, and the similarity measured for them.
CAPITAL = ("What is the capital of France?", "Tell me France's capital city.", 0.9026)
REFUND = ("What is your refund policy?", "How do I get my money back?", 0.5510)
ENTITY_SWAP = ("How do I cancel my order?", "How do I cancel my account?", 0.6816)
NUMERIC_CHANGE = ("What is 15 percent of 200?", "What is 25 percent of 200?", 0.7904)
NEGATION = ("Is the store open on Sunday?", "Is the store not open on Sunday?", 0.9716)


@pytest.fixture(scope="module")
def real_embedder() -> OnnxEmbedder:
    """The real model, loaded once for the module because loading is the slow part."""
    embedder = OnnxEmbedder(SUPPORTED_MODEL)
    embedder.load()
    return embedder


async def similarity(embedder: OnnxEmbedder, pair: tuple[str, str, float]) -> float:
    original, rewrite, _ = pair
    return cosine(await embedder.encode(original), await embedder.encode(rewrite))


# --- the measured scores ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pair"),
    [
        ("a paraphrase the model handles well", CAPITAL),
        ("the PRD's own paraphrase example", REFUND),
        ("an entity swap", ENTITY_SWAP),
        ("a changed number", NUMERIC_CHANGE),
        ("a negation", NEGATION),
    ],
)
async def test_the_measured_similarity_has_not_moved(
    real_embedder: OnnxEmbedder, name: str, pair: tuple[str, str, float]
) -> None:
    measured = await similarity(real_embedder, pair)
    assert measured == pytest.approx(pair[2], abs=TOLERANCE), (
        f"{name} now scores {measured:.4f}, was {pair[2]}; PRD section 33.4 needs revisiting"
    )


@pytest.mark.parametrize("threshold", SWEPT_THRESHOLDS)
async def test_no_swept_threshold_gets_both_the_paraphrase_and_the_negation_right(
    real_embedder: OnnxEmbedder, threshold: float
) -> None:
    """The project's central finding, at every threshold the benchmark sweeps.

    The PRD's flagship paraphrase scores below the threshold, so it misses, while
    the negation scores above it, so it hits and serves the opposite answer.
    Raising the threshold cannot fix this, because the negation outscores the
    paraphrase; lowering it only adds false hits.
    """
    paraphrase = await similarity(real_embedder, REFUND)
    negation = await similarity(real_embedder, NEGATION)

    assert paraphrase < threshold <= negation


# --- the same facts through the real request path -------------------------------------


@pytest.fixture
def tier2(
    client: TestClient,
    real_embedder: OnnxEmbedder,
    chroma_factory: Callable[[], Any],
    collection_name: str,
) -> Callable[[float], TestClient]:
    """A gateway whose Tier 2 is the real model over in-memory Chroma, at a chosen threshold."""

    def configure(threshold: float) -> TestClient:
        client.app.state.semantic_cache = SemanticCache(  # type: ignore[attr-defined]
            chroma_factory,
            real_embedder,
            collection_name=collection_name,
            ttl_seconds=3600,
            threshold=threshold,
        )
        return client

    return configure


def ask(client: TestClient, content: str) -> dict[str, Any]:
    response = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
        headers=AUTH,
    )
    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    return payload


def test_a_paraphrase_above_the_threshold_is_a_semantic_hit_with_its_score(
    tier2: Callable[[float], TestClient], stub_provider: StubProvider
) -> None:
    """TEST-004."""
    client = tier2(0.85)
    original, rewrite, expected = CAPITAL

    ask(client, original)
    reply = ask(client, rewrite)

    assert reply["semcache_status"] == "semantic_hit"
    assert reply["semcache_similarity"] == pytest.approx(expected, abs=TOLERANCE)
    assert stub_provider.call_count == 1


def test_a_paraphrase_below_the_threshold_reaches_the_provider(
    tier2: Callable[[float], TestClient], stub_provider: StubProvider
) -> None:
    """TEST-005. The same pair as above, with the threshold raised past its score."""
    client = tier2(0.95)
    original, rewrite, _ = CAPITAL

    ask(client, original)
    reply = ask(client, rewrite)

    assert reply["semcache_status"] == "miss"
    assert reply["semcache_similarity"] is None
    assert stub_provider.call_count == 2


def test_the_prds_own_paraphrase_example_misses_even_at_the_lowest_swept_threshold(
    tier2: Callable[[float], TestClient], stub_provider: StubProvider
) -> None:
    client = tier2(min(SWEPT_THRESHOLDS))
    original, rewrite, _ = REFUND

    ask(client, original)

    assert ask(client, rewrite)["semcache_status"] == "miss"
    assert stub_provider.call_count == 2


def test_a_negated_question_is_served_the_opposite_answer(
    tier2: Callable[[float], TestClient], stub_provider: StubProvider
) -> None:
    """TEST-006, as the finding it actually is: this hits, and it should not.

    PRD section 33.4 explains why this asserts the false hit instead of a miss.
    If it starts failing because the negation now misses, that is good news about
    the model, and the PRD and the evaluation report have to be updated to say so.
    """
    client = tier2(0.90)
    original, rewrite, _ = NEGATION

    first = ask(client, original)
    served = ask(client, rewrite)

    assert served["semcache_status"] == "semantic_hit"
    assert served["choices"][0]["message"] == first["choices"][0]["message"]
    assert stub_provider.call_count == 1


def test_a_swapped_entity_misses_at_the_default_threshold(
    tier2: Callable[[float], TestClient], stub_provider: StubProvider
) -> None:
    """TEST-007."""
    client = tier2(0.90)
    original, rewrite, _ = ENTITY_SWAP

    ask(client, original)

    assert ask(client, rewrite)["semcache_status"] == "miss"
    assert stub_provider.call_count == 2


@pytest.mark.parametrize(("threshold", "expected"), [(0.90, "miss"), (0.75, "semantic_hit")])
def test_a_changed_number_misses_at_the_default_threshold_but_not_the_lowest(
    tier2: Callable[[float], TestClient], threshold: float, expected: str
) -> None:
    """TEST-008, and the threshold it depends on: 15 percent and 25 percent score 0.79."""
    client = tier2(threshold)
    original, rewrite, _ = NUMERIC_CHANGE

    ask(client, original)

    assert ask(client, rewrite)["semcache_status"] == expected
