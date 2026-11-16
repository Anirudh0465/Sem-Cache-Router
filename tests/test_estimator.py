# Token estimator tests.
#
# Most use a stand in encoding that counts words, so the arithmetic is checkable by
# eye and no vocabulary is downloaded. Two things need the real library and still
# not the network: that special token text in a prompt is counted rather than
# rejected, and that counts come from the encoding rather than from characters.
# For those, a real tiktoken Encoding is built from a tiny byte level vocabulary,
# which tiktoken accepts directly. One test against the real cl100k_base is marked
# model, because loading it downloads a vocabulary.

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
import tiktoken

from app.limiter.estimator import (
    OFF_LOOP_CHARACTERS,
    REPLY_PRIMING,
    TOKENS_PER_MESSAGE,
    TokenEstimator,
)
from app.models import Message

DEFAULT_REPLY = 256


class WordEncoding:
    """Counts whitespace separated words, and records where it was asked to run."""

    def __init__(self) -> None:
        self.threads: list[int] = []
        self.disallowed: list[Any] = []

    def encode(self, text: str, *, disallowed_special: tuple[()] = ()) -> list[int]:
        self.threads.append(threading.get_ident())
        self.disallowed.append(disallowed_special)
        return list(range(len(text.split())))


def loaded(encoding: Any = None) -> TokenEstimator:
    estimator = TokenEstimator(loader=lambda name: encoding or WordEncoding())
    estimator.load()
    return estimator


def byte_level_encoding() -> tiktoken.Encoding:
    """A real tiktoken encoding with one token per byte and one special token."""
    return tiktoken.Encoding(
        name="byte-level-for-tests",
        pat_str=r"(?s:.)",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )


def user(content: str) -> Message:
    return Message(role="user", content=content)


# --- counting -------------------------------------------------------------------------


def test_counts_come_from_the_loaded_encoding() -> None:
    assert loaded().count("one two three") == 3


def test_before_loading_the_character_rule_is_used() -> None:
    estimator = TokenEstimator(loader=lambda name: WordEncoding())

    assert estimator.exact is False
    assert estimator.count("x" * 400) == 100
    assert estimator.count("") == 1, "even an empty message costs something"


def test_loading_makes_the_estimator_exact_and_asks_for_the_named_encoding() -> None:
    asked: list[str] = []

    def loader(name: str) -> WordEncoding:
        asked.append(name)
        return WordEncoding()

    estimator = TokenEstimator(loader=loader, encoding="cl100k_base")
    estimator.load()

    assert estimator.exact is True
    assert asked == ["cl100k_base"]


def test_a_failed_load_falls_back_without_raising_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Offline, or the vocabulary host down: the gateway must still serve."""

    def broken(name: str) -> Any:
        raise OSError("no route to host")

    estimator = TokenEstimator(loader=broken)
    with caplog.at_level(logging.WARNING, logger="app.limiter.estimator"):
        estimator.load()

    assert estimator.exact is False
    assert estimator.count("x" * 400) == 100
    assert "fall back to four characters" in caplog.text


def test_special_token_text_in_a_prompt_is_counted_not_rejected() -> None:
    """tiktoken raises on these by default, and a caller can put them in a prompt."""
    encoding = byte_level_encoding()
    with pytest.raises(ValueError):
        encoding.encode("<|endoftext|>")

    assert loaded(encoding).count("<|endoftext|>") == len("<|endoftext|>")


def test_the_estimator_asks_for_special_tokens_to_be_allowed_as_text() -> None:
    encoding = WordEncoding()
    loaded(encoding).count("anything")

    assert encoding.disallowed == [()]


def test_text_that_is_dense_in_tokens_is_not_undercounted_by_the_character_rule() -> None:
    """Three characters, nine bytes: the rule that says one token would let a caller
    reserve a ninth of what such text costs."""
    estimator = loaded(byte_level_encoding())

    assert estimator.count("日本語") == 9
    assert TokenEstimator(loader=lambda name: WordEncoding()).count("日本語") == 1


# --- estimating a request -------------------------------------------------------------


async def test_the_estimate_is_the_framed_prompt_plus_the_reply() -> None:
    """One message: 3 of framing, 1 for the role, 3 words, 3 priming, then the reply."""
    estimator = loaded()

    estimate = await estimator.estimate(
        [user("one two three")], max_tokens=50, default_completion_tokens=DEFAULT_REPLY
    )

    assert estimate == TOKENS_PER_MESSAGE + 1 + 3 + REPLY_PRIMING + 50


async def test_each_message_carries_its_own_framing() -> None:
    estimator = loaded()
    messages = [Message(role="system", content="be brief"), user("one two three")]

    estimate = await estimator.estimate(
        messages, max_tokens=0, default_completion_tokens=DEFAULT_REPLY
    )

    assert estimate == 2 * (TOKENS_PER_MESSAGE + 1) + 2 + 3 + REPLY_PRIMING


async def test_the_default_allowance_covers_a_reply_the_request_did_not_bound() -> None:
    estimator = loaded()

    unbounded = await estimator.estimate(
        [user("hi")], max_tokens=None, default_completion_tokens=DEFAULT_REPLY
    )
    bounded = await estimator.estimate(
        [user("hi")], max_tokens=10, default_completion_tokens=DEFAULT_REPLY
    )

    assert unbounded - bounded == DEFAULT_REPLY - 10


async def test_an_explicit_zero_is_respected_and_does_not_fall_back_to_the_default() -> None:
    estimator = loaded()

    zero = await estimator.estimate(
        [user("hi")], max_tokens=0, default_completion_tokens=DEFAULT_REPLY
    )

    assert zero == TOKENS_PER_MESSAGE + 1 + 1 + REPLY_PRIMING


async def test_the_estimate_works_before_the_encoding_is_loaded() -> None:
    estimator = TokenEstimator(loader=lambda name: WordEncoding())

    estimate = await estimator.estimate(
        [user("x" * 400)], max_tokens=0, default_completion_tokens=DEFAULT_REPLY
    )

    assert estimate == TOKENS_PER_MESSAGE + 1 + 100 + REPLY_PRIMING


# --- where the work runs --------------------------------------------------------------


async def test_a_short_prompt_is_counted_on_the_event_loop() -> None:
    encoding = WordEncoding()

    await loaded(encoding).estimate(
        [user("a short prompt")], max_tokens=None, default_completion_tokens=DEFAULT_REPLY
    )

    assert set(encoding.threads) == {threading.get_ident()}


async def test_a_long_prompt_is_counted_off_the_event_loop() -> None:
    """A megabyte of text would otherwise hold every other request while it was encoded."""
    encoding = WordEncoding()

    await loaded(encoding).estimate(
        [user("word " * OFF_LOOP_CHARACTERS)],
        max_tokens=None,
        default_completion_tokens=DEFAULT_REPLY,
    )

    assert threading.get_ident() not in encoding.threads
    assert encoding.threads, "it was counted somewhere"


# --- the real vocabulary --------------------------------------------------------------


@pytest.mark.model
@pytest.mark.parametrize(
    ("text", "tokens"),
    [("What is the capital of France?", 7), ("x" * 400, 50), ("日本語のテキスト", 8)],
)
def test_the_real_encoding_counts_what_the_character_rule_gets_wrong(
    text: str, tokens: int
) -> None:
    """Opt in with SEMCACHE_RUN_MODEL_TESTS=1; the first load downloads a vocabulary.

    400 repeated characters is 50 tokens where four characters a token says 100, and
    eight Japanese characters are 8 where it says 2.
    """
    estimator = TokenEstimator()
    estimator.load()

    assert estimator.exact
    assert estimator.count(text) == tokens
