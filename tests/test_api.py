# Route level tests for wire compatibility and error mapping.
#
# Will cover:
#   an unmodified OpenAI client works against the gateway with only a base URL
#   change
#   the additive response fields are ignorable by a client that does not know
#   them
#   an invalid body returns 400
#   a missing API key returns 401, since the key is the budget identity
#   all providers down returns 502
#   /health reports dependency reachability, so a degraded gateway is
#   distinguishable from a stopped one
#   the cache flush endpoint empties both tiers
#   /metrics and /v1/cache/stats agree with each other
