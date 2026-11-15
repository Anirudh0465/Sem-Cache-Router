# Rate limiter script tests.
#
# These drive the two Lua scripts directly, on a fake Redis that runs real Lua, so
# what is tested is the code that runs in production and not a model of it. They
# cover the property the limiter exists for, that concurrent callers cannot spend
# the same tokens twice, and the arithmetic around it: refill, the retry hint, the
# clamps, and what a request too large for any bucket does.
#
# The reserve script reads the clock from Redis. Most tests pass a time in instead,
# the optional fourth argument, so refill is exact and no test sleeps; two tests
# leave it out, to prove the path production takes.

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
from fakeredis import aioredis

LIMITER = Path(__file__).resolve().parents[1] / "app" / "limiter"
ADMITTED, DENIED, TOO_LARGE = 1, 0, 2

CAPACITY = 100
RATE = 10
KEY = "rl:caller:tokens"


class Limiter:
    """The two scripts registered against a Redis, with the arguments spelled out."""

    def __init__(self, redis: aioredis.FakeRedis) -> None:
        self.redis = redis
        self._reserve = redis.register_script((LIMITER / "scripts.lua").read_text())
        self._adjust = redis.register_script((LIMITER / "adjust.lua").read_text())

    async def reserve(
        self,
        estimate: float,
        *,
        now: float | None = None,
        key: str = KEY,
        capacity: float = CAPACITY,
        rate: float = RATE,
    ) -> tuple[int, int, int]:
        args = [capacity, rate, estimate] + ([] if now is None else [now])
        status, remaining, retry = await self._reserve(keys=[key], args=args)
        return int(status), int(remaining), int(retry)

    async def adjust(self, delta: float, *, key: str = KEY, capacity: float = CAPACITY) -> int:
        return int(await self._adjust(keys=[key], args=[capacity, delta]))


@pytest.fixture
def limiter(fake_redis: aioredis.FakeRedis) -> Limiter:
    return Limiter(fake_redis)


# --- reserving ------------------------------------------------------------------------


async def test_a_new_bucket_starts_full(limiter: Limiter) -> None:
    assert await limiter.reserve(60, now=0) == (ADMITTED, 40, 0)


async def test_a_request_that_fits_exactly_is_admitted(limiter: Limiter) -> None:
    assert await limiter.reserve(CAPACITY, now=0) == (ADMITTED, 0, 0)


async def test_a_request_that_does_not_fit_is_denied_and_spends_nothing(limiter: Limiter) -> None:
    await limiter.reserve(60, now=0)

    status, remaining, retry = await limiter.reserve(50, now=0)

    assert (status, remaining) == (DENIED, 40)
    assert retry > 0
    assert await limiter.reserve(40, now=0) == (ADMITTED, 0, 0), "the denial must not have cost 40"


@pytest.mark.parametrize(("estimate", "retry"), [(50, 1), (55, 2), (100, 6)])
async def test_the_retry_hint_is_the_whole_seconds_the_shortfall_takes_to_refill(
    limiter: Limiter, estimate: int, retry: int
) -> None:
    """With 40 tokens left and 10 restored a second: a shortfall of 10, 15 and 60."""
    await limiter.reserve(60, now=0)

    assert (await limiter.reserve(estimate, now=0))[2] == retry


async def test_tokens_refill_with_time(limiter: Limiter) -> None:
    await limiter.reserve(CAPACITY, now=0)

    assert await limiter.reserve(30, now=3) == (ADMITTED, 0, 0)
    assert (await limiter.reserve(1, now=3))[0] == DENIED


async def test_refill_is_not_rounded_to_whole_seconds(limiter: Limiter) -> None:
    await limiter.reserve(CAPACITY, now=0)

    assert await limiter.reserve(5, now=0.5) == (ADMITTED, 0, 0)


async def test_refill_stops_at_capacity(limiter: Limiter) -> None:
    await limiter.reserve(10, now=0)

    assert await limiter.reserve(CAPACITY, now=100_000) == (ADMITTED, 0, 0)
    assert (await limiter.reserve(1, now=100_000))[0] == DENIED


async def test_a_denial_does_not_make_the_next_caller_pay_for_refill_twice(
    limiter: Limiter,
) -> None:
    """The refill counted at a denial is written back, so it is not counted again."""
    await limiter.reserve(CAPACITY, now=0)
    assert (await limiter.reserve(50, now=1))[0] == DENIED

    assert await limiter.reserve(20, now=2) == (ADMITTED, 0, 0)
    assert (await limiter.reserve(1, now=2))[0] == DENIED


async def test_a_clock_that_runs_backwards_gives_no_refill_and_no_debt(limiter: Limiter) -> None:
    await limiter.reserve(60, now=100)

    assert await limiter.reserve(10, now=50) == (ADMITTED, 30, 0)


async def test_buckets_are_independent(limiter: Limiter) -> None:
    await limiter.reserve(CAPACITY, now=0, key="rl:a:tokens")

    assert (await limiter.reserve(1, now=0, key="rl:a:tokens"))[0] == DENIED
    assert await limiter.reserve(CAPACITY, now=0, key="rl:b:tokens") == (ADMITTED, 0, 0)


async def test_a_bucket_expires_once_it_would_have_refilled(
    limiter: Limiter, fake_redis: aioredis.FakeRedis
) -> None:
    """Capacity 100 at 10 a second is full again after 10 seconds, so that is its life."""
    await limiter.reserve(60, now=0)

    assert await fake_redis.ttl(KEY) == math.ceil(CAPACITY / RATE)


# --- a request nothing can satisfy ----------------------------------------------------


async def test_a_request_larger_than_the_whole_bucket_is_never_admissible(
    limiter: Limiter, fake_redis: aioredis.FakeRedis
) -> None:
    assert await limiter.reserve(CAPACITY + 1, now=0) == (TOO_LARGE, 0, 0)
    assert await fake_redis.exists(KEY) == 0, "refusing it must not create a bucket"


async def test_a_request_larger_than_the_bucket_leaves_an_existing_balance_alone(
    limiter: Limiter,
) -> None:
    await limiter.reserve(60, now=0)

    assert (await limiter.reserve(CAPACITY + 1, now=0))[0] == TOO_LARGE
    assert await limiter.reserve(40, now=0) == (ADMITTED, 0, 0)


# --- the property the limiter exists for ----------------------------------------------


async def test_concurrent_callers_cannot_spend_the_same_tokens(limiter: Limiter) -> None:
    """TEST-011. Twenty at once for ten tokens each, with room for ten of them.

    The refill rate is negligible, so nothing can be earned while they run.
    """
    results = await asyncio.gather(*[limiter.reserve(10, rate=0.0001) for _ in range(20)])

    admitted = [r for r in results if r[0] == ADMITTED]
    assert len(admitted) == 10
    assert sorted(r[1] for r in admitted) == list(range(0, 100, 10))
    assert all(r[0] == DENIED for r in results if r not in admitted)


async def test_time_comes_from_redis_when_the_caller_gives_none(limiter: Limiter) -> None:
    """The production path: no clock passed in, so refill is Redis's own.

    Drain a bucket that refills a thousand tokens a second and wait a tenth of one.
    A script that never read the clock would see no time pass and refuse the second
    request, so this fails if the real clock is not being used.
    """
    assert await limiter.reserve(CAPACITY, rate=1000) == (ADMITTED, 0, 0)

    await asyncio.sleep(0.1)

    assert (await limiter.reserve(30, rate=1000))[0] == ADMITTED


async def test_a_clock_the_caller_does_pass_is_the_one_used(limiter: Limiter) -> None:
    """Both sources work, and the explicit one wins, which is what lets tests set the time."""
    await limiter.reserve(CAPACITY, now=0, rate=1000)

    assert (await limiter.reserve(30, now=0, rate=1000))[0] == DENIED


# --- settling a reservation -----------------------------------------------------------


async def test_a_credit_returns_tokens(limiter: Limiter) -> None:
    """TEST-012. Reserved 60, used 30: the difference goes back."""
    await limiter.reserve(60, now=0)

    assert await limiter.adjust(+30) == 70


async def test_returning_a_whole_reservation_restores_the_bucket(limiter: Limiter) -> None:
    """A cache hit spends nothing, so all of what was reserved comes back."""
    await limiter.reserve(60, now=0)

    assert await limiter.adjust(+60) == CAPACITY


async def test_a_credit_cannot_take_a_bucket_past_capacity(limiter: Limiter) -> None:
    await limiter.reserve(10, now=0)

    assert await limiter.adjust(+1000) == CAPACITY


async def test_a_debit_takes_tokens_for_a_reply_that_cost_more_than_reserved(
    limiter: Limiter,
) -> None:
    await limiter.reserve(60, now=0)

    assert await limiter.adjust(-30) == 10


async def test_a_debit_cannot_leave_a_bucket_in_debt(limiter: Limiter) -> None:
    await limiter.reserve(60, now=0)

    assert await limiter.adjust(-1000) == 0


async def test_settling_a_bucket_that_has_expired_does_nothing(
    limiter: Limiter, fake_redis: aioredis.FakeRedis
) -> None:
    """It expired, so it is full, so there is nothing to credit and nothing to invent."""
    assert await limiter.adjust(+50) == -1
    assert await fake_redis.exists(KEY) == 0


async def test_settling_does_not_extend_the_life_or_move_the_refill_time(
    limiter: Limiter, fake_redis: aioredis.FakeRedis
) -> None:
    await limiter.reserve(60, now=42)
    await fake_redis.expire(KEY, 3)

    await limiter.adjust(+10)

    assert await fake_redis.ttl(KEY) <= 3
    assert float(await fake_redis.hget(KEY, "last_refill_at")) == 42.0
