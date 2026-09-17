# Per provider circuit breaker.
#
# Will hold:
#   a BreakerState enum of CLOSED, OPEN and HALF_OPEN
#   a CircuitBreaker class with call, _on_success, _on_failure, _backoff and a
#   helper that decides whether an exception counts as a failure
#
# State transitions:
#   CLOSED     passes traffic and counts failures
#   OPEN       rejects immediately, without attempting a call
#   HALF_OPEN  allows a limited number of probes after the backoff expires
#   a probe that succeeds closes the breaker
#   a probe that fails reopens it with a longer backoff
#
# Backoff shape:
#   delay  = min(base * 2 ** attempt, ceiling)
#   actual = random.uniform(0, delay)
#
# Why jitter: without it, every client that failed at the same moment retries at
# the same moment, and a recovering provider receives a synchronised burst
# precisely when it is least able to absorb one. Jitter turns a thundering herd
# into a gradual ramp.
#
# Failure classification is deliberate. Timeouts, 5xx responses and provider
# 429s count toward the breaker. A 400 does not, because a malformed request is
# a caller defect, and counting it would let one badly behaved client trip
# failover for everybody.
