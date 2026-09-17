# Token bucket tests.
#
# Will cover:
#   TEST-010  a caller over budget receives 429, cached or not
#   TEST-011  concurrent reserves against one nearly empty bucket cannot both
#             be admitted
#   TEST-012  actual usage below the estimate returns the difference
#   a cache hit releases the reservation in full
#   the retry hint is derived from the refill rate rather than guessed
#   TEST-016  with Redis unreachable, admission cannot be decided, so 503
