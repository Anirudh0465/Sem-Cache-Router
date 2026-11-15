-- Atomic refill and reserve for the token bucket.
--
-- Refilling and spending are one script because Redis runs a script without
-- interleaving anything else. Two callers arriving together therefore cannot both
-- read the same balance and both be admitted for tokens only one of them can have,
-- which is the whole property the limiter exists to provide.
--
-- KEYS[1]  bucket key, of the form rl:{api_key}:{window}
-- ARGV[1]  capacity, the most a bucket can hold
-- ARGV[2]  refill_rate, tokens restored per second
-- ARGV[3]  estimate, the tokens to reserve
-- ARGV[4]  now, optional: the time to use, in seconds
--
-- Returns three numbers:
--   status     1 admitted, 0 denied for now, 2 never admissible
--   remaining  whole tokens left in the bucket, rounded down
--   retry      whole seconds until the request could be admitted, 0 unless denied
--
-- Time comes from Redis, not from the caller. With several gateways in front of
-- one Redis, a clock passed in by each would let a gateway whose clock runs ahead
-- hand its callers refill that no other gateway agrees they have earned. ARGV[4]
-- exists so a test can set the clock, and production never passes it.
--
-- A request larger than the whole bucket is status 2 rather than a denial. No
-- amount of waiting makes it fit, and a retry hint on it would send the caller
-- back every few seconds for ever.

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local estimate = tonumber(ARGV[3])

local now = tonumber(ARGV[4])
if now == nil then
    local time = redis.call("TIME")
    now = tonumber(time[1]) + tonumber(time[2]) / 1000000
end

if estimate > capacity then
    return {2, 0, 0}
end

local bucket = redis.call("HMGET", key, "tokens_remaining", "last_refill_at")
local tokens = tonumber(bucket[1])
local last_refill_at = tonumber(bucket[2])

-- A bucket that does not exist, or has expired, is full: that is what expiry means.
if tokens == nil or last_refill_at == nil then
    tokens = capacity
    last_refill_at = now
end

local elapsed = math.max(0, now - last_refill_at)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local status = 0
local retry = 0
if tokens >= estimate then
    tokens = tokens - estimate
    status = 1
else
    retry = math.ceil((estimate - tokens) / refill_rate)
end

-- State is written on a denial too, so the refill just counted is not counted again
-- by the next caller. Expiry at the time a drained bucket takes to refill is
-- equivalent to never expiring: by then it is full, and a missing bucket is full.
redis.call("HSET", key, "tokens_remaining", tokens, "last_refill_at", now)
redis.call("EXPIRE", key, math.max(1, math.ceil(capacity / refill_rate)))

return {status, math.floor(tokens), retry}
