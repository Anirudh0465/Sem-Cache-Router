# Circuit breaker tests.
#
# Will cover:
#   TEST-013  repeated primary failures open the breaker
#   traffic moves to the secondary with no caller visible error
#   TEST-014  a successful half open probe closes the breaker
#   a failed probe reopens it with a longer backoff, not a reset one
#   a 400 does not count as a failure
#   timeouts, 5xx and provider 429s do count
#   two consecutive backoffs differ, or the herd stays synchronised
#   exponential growth respects the ceiling
