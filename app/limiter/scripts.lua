-- Atomic refill and reserve for the token bucket.
--
-- KEYS[1]  bucket key, of the form rl:{api_key}:{window}
-- ARGV[1]  capacity, the maximum balance
-- ARGV[2]  refill_rate, tokens restored per second
-- ARGV[3]  estimate, the tokens to reserve
-- ARGV[4]  now, a unix timestamp
--
-- Returns:  admitted flag, tokens remaining, retry after seconds

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local estimate = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local bucket = redis.call("HMGET", key, "tokens_remaining", "last_refill_at")
local tokens_remaining = tonumber(bucket[1])
local last_refill_at = tonumber(bucket[2])

if tokens_remaining == nil then
    tokens_remaining = capacity
    last_refill_at = now
end

local elapsed = now - last_refill_at
if elapsed > 0 then
    local new_tokens = elapsed * refill_rate
    tokens_remaining = math.min(capacity, tokens_remaining + new_tokens)
    last_refill_at = now
end

if tokens_remaining >= estimate then
    tokens_remaining = tokens_remaining - estimate
    redis.call("HMSET", key, "tokens_remaining", tokens_remaining, "last_refill_at", last_refill_at)
    -- Expire after window? The python code handles window in key name, but we can set TTL here.
    -- If refill rate is e.g. 1 per second, and capacity is 100, full refill is 100s.
    local ttl = math.ceil(capacity / refill_rate)
    redis.call("EXPIRE", key, ttl)
    return {1, tokens_remaining, 0}
else
    local needed = estimate - tokens_remaining
    local retry_after = math.ceil(needed / refill_rate)
    redis.call("HMSET", key, "tokens_remaining", tokens_remaining, "last_refill_at", last_refill_at)
    local ttl = math.ceil(capacity / refill_rate)
    redis.call("EXPIRE", key, ttl)
    return {0, tokens_remaining, retry_after}
end
