# Tier 1: exact match cache in Redis.
#
# Will hold an ExactCache class with:
#   get    look up a prompt, returning the entry or nothing on a miss
#   set    write a provider response with a fresh TTL
#   _key   derive the Redis key, delegating to the shared hasher
#
# A hit updates hit count and last hit timestamp but leaves the remaining TTL
# untouched. TTL here means freshness, not popularity, and a sliding expiry
# would let a frequently requested answer live indefinitely, which is exactly
# where a stale answer does the most damage.
#
# Writes happen only on a provider miss, so the cost is paid once per genuinely
# new answer.
