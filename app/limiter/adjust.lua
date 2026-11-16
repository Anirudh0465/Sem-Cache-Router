-- Credit or debit a bucket by a signed amount, clamped to what a bucket can hold.
--
-- One script for both directions of settling a reservation. A cache hit returns
-- the whole estimate, and a provider call returns the difference when the reply
-- cost less than was reserved, or takes the difference when it cost more. In each
-- case the answer is the same arithmetic with a different sign.
--
-- KEYS[1]  bucket key, of the form rl:{api_key}:{window}
-- ARGV[1]  capacity, the most a bucket can hold
-- ARGV[2]  delta, positive to credit tokens and negative to debit them
--
-- Returns the whole tokens now in the bucket, rounded down, or -1 if there is no
-- bucket.
--
-- The balance is clamped to between zero and capacity. A debit cannot take a
-- bucket below zero, so a reply that cost more than was reserved is absorbed
-- rather than leaving the caller in debt; the overshoot is not recovered.
--
-- A missing bucket is left missing. It expired, and an expired bucket is full, so
-- there is nothing to credit, and creating one here would invent state for a
-- caller who has none. The expiry and the refill time of an existing bucket are
-- left alone: settling a reservation is not activity that should delay either.

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local delta = tonumber(ARGV[2])

local current = redis.call("HGET", key, "tokens_remaining")
if current == false then
    return -1
end

local updated = math.min(capacity, math.max(0, tonumber(current) + delta))
redis.call("HSET", key, "tokens_remaining", updated)

return math.floor(updated)
