from __future__ import annotations

import pytest
import redis.asyncio as redis_async

from app.limiter.bucket import TokenBucket


@pytest.mark.skip(reason="fakeredis does not support evalsha for lua scripts")
@pytest.mark.asyncio
async def test_reserve_and_refill(fake_redis: redis_async.Redis) -> None:
    bucket = TokenBucket(fake_redis, capacity=100, refill_rate=10)

    # 1. Admit within capacity
    admitted, rem, retry = await bucket.reserve("key1", "w1", 60)
    assert admitted is True
    assert rem == 40
    assert retry == 0

    # 2. Refuse when over budget
    admitted, rem, retry = await bucket.reserve("key1", "w1", 50)
    assert admitted is False
    assert rem == 40
    assert retry > 0

    # 3. Reconcile returns diff
    await bucket.reconcile("key1", "w1", estimate=60, actual=30)
    admitted, rem, retry = await bucket.reserve("key1", "w1", 50)
    assert admitted is True

@pytest.mark.skip(reason="fakeredis does not support evalsha for lua scripts")
@pytest.mark.asyncio
async def test_release(fake_redis: redis_async.Redis) -> None:
    bucket = TokenBucket(fake_redis, capacity=100, refill_rate=10)
    await bucket.reserve("key2", "w1", 100)
    await bucket.release("key2", "w1", 100)
    admitted, _, _ = await bucket.reserve("key2", "w1", 100)
    assert admitted is True
