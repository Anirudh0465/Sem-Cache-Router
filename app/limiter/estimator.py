# What a request will cost, in provider tokens, before the provider has answered.
#
# The limiter has to decide whether to admit a request before the reply exists, so
# it reserves an estimate and settles against reported usage afterwards (PRD section
# 16). The estimate is the prompt, counted with tiktoken, plus what the reply may
# be: the request's own max_tokens if it set one, otherwise a configured allowance.
# Counting only the prompt would admit a caller for far less than a long answer
# costs, and a count that was wrong in the caller's favour is a hole in the budget.
#
# Characters divided by four, which the provider adapters use today, is wrong in
# both directions: repeated text is 8 characters to a token and Japanese is 1, so a
# caller sending non Latin text could reserve a quarter of what it spends.
#
# The vocabulary tiktoken needs is downloaded the first time an encoding is loaded,
# and that took 18 seconds where this was written. So loading is a separate step,
# done at startup, and never happens on a request. If it fails, because the machine
# is offline or the host is down, the estimator falls back to the old rule and says
# so in the log: an estimate that is less accurate is better than a limiter that
# stops the gateway serving.
#
# One encoding is used for every model. Anthropic tokenises differently, so for its
# models this is an approximation, which is acceptable for a reservation that is
# reconciled against the provider's own count once the call completes.

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from app.models import Message

log = logging.getLogger(__name__)

ENCODING = "cl100k_base"

# The framing OpenAI's chat format wraps around messages, from its published
# counting guidance: a few tokens of markup per message, and three that prime the
# reply. Small, but a limiter that ignores it is wrong on every short request.
TOKENS_PER_MESSAGE = 3
REPLY_PRIMING = 3

# Text longer than this is counted in a worker thread. Encoding is CPU bound, and a
# request body can be a megabyte, which would hold the event loop for every other
# request while it ran.
OFF_LOOP_CHARACTERS = 20_000


class Encoding(Protocol):
    """The part of a tiktoken encoding the estimator uses."""

    def encode(self, text: str, *, disallowed_special: tuple[()] = ()) -> list[int]: ...


EncodingLoader = Callable[[str], Encoding]


def _load_tiktoken(name: str) -> Encoding:
    """Load a real encoding. Imported here so nothing else pays for it, or needs it."""
    import tiktoken

    return tiktoken.get_encoding(name)


class TokenEstimator:
    """Counts tokens, and estimates the cost of a request."""

    def __init__(self, loader: EncodingLoader = _load_tiktoken, encoding: str = ENCODING) -> None:
        self._loader = loader
        self._encoding_name = encoding
        self._encoding: Encoding | None = None

    @property
    def exact(self) -> bool:
        """Whether counts come from tiktoken, or from the character rule it falls back to."""
        return self._encoding is not None

    def load(self) -> None:
        """Load the encoding. Blocking, so call it off the event loop and at startup.

        Never raises. A failure leaves the estimator on the fallback rule and is
        logged, because the alternative is a gateway that will not start, or one that
        refuses every request, for want of a vocabulary download.
        """
        try:
            self._encoding = self._loader(self._encoding_name)
        except Exception:
            log.warning(
                "could not load the %s encoding; token estimates fall back to four "
                "characters a token, which is less accurate",
                self._encoding_name,
                exc_info=True,
            )

    def count(self, text: str) -> int:
        """Tokens in a piece of text.

        Special token strings in the text, such as the end of text marker, are
        counted as the ordinary characters they are. tiktoken's default is to raise,
        and a caller can put those characters in a prompt, so the default would turn
        a prompt into an error.
        """
        if self._encoding is None:
            return max(1, len(text) // 4)
        return len(self._encoding.encode(text, disallowed_special=()))

    def _prompt_tokens(self, messages: Sequence[Message]) -> int:
        framed = sum(
            TOKENS_PER_MESSAGE + self.count(message.role) + self.count(message.content)
            for message in messages
        )
        return framed + REPLY_PRIMING

    async def estimate(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None,
        default_completion_tokens: int,
    ) -> int:
        """The tokens to reserve: the prompt as the provider will count it, plus the reply.

        The reply allowance is the request's max_tokens when it sets one, since that
        is the most it can cost, and the configured default when it does not.
        """
        reply = default_completion_tokens if max_tokens is None else max_tokens
        characters = sum(len(message.content) for message in messages)
        if characters > OFF_LOOP_CHARACTERS:
            prompt = await asyncio.to_thread(self._prompt_tokens, messages)
        else:
            prompt = self._prompt_tokens(messages)
        return prompt + reply
