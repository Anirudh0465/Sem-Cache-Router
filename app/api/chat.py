# Chat completions and cache administration routes.
#
# Will hold:
#   POST /v1/chat/completions   the primary endpoint
#   DELETE /v1/cache            flush both tiers, to force a cold benchmark start
#   GET /v1/cache/stats         hit ratio by tier, entry count, eviction count
#
# The completions handler owns the ordering the whole design rests on:
#   1. validate the body against the chat completion schema
#   2. normalise the prompt and estimate its token cost with tiktoken
#   3. reserve that estimate from the caller bucket, or return 429 and stop
#   4. check Tier 1, and on a hit release the reservation in full
#   5. on a Tier 1 miss, embed and search Tier 2 against theta
#   6. on a Tier 2 miss, route a model and call the provider through the breaker
#   7. on provider failure, let the breaker move traffic to the secondary
#   8. write both tiers with a TTL, reconcile the reservation, return
#
# Why the limiter runs before any cache lookup: a budget that depends on what
# happens to be cached is not a budget anyone can reason about. The accepted
# cost is that a caller at their limit is refused an answer that was free to
# serve.
