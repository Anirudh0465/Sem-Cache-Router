# Redis backed token bucket.
#
# Will hold a TokenBucket class with:
#   load_script  register the Lua script and cache its SHA at startup
#   reserve      hold an estimated token cost, or refuse when over budget
#   reconcile    settle a reservation against reported usage
#   release      return a reservation in full, used on a cache hit
#   retry_after  seconds until admission is possible, from the refill rate
#
# Why the unit is provider tokens rather than requests: providers enforce token
# limits, and a request count limit does nothing to protect against a small
# number of very long prompts.
#
# Why admission uses an estimate: the true cost of a response is not known until
# the provider replies. tiktoken estimates the prompt plus the requested
# completion length, that estimate is reserved, and the error is corrected at
# reconciliation rather than left to drift.
#
# Why release exists separately from reconcile: a cache hit spends no provider
# tokens at all, so the whole reservation comes back.
