# Counters exposed on /metrics.
#
# Will hold a Metrics class recording:
#   cache hits, kept separate per tier since the two mean different things
#   misses that reached a provider
#   throttles, which are a success for the limiter rather than a failure
#   breaker driven failovers, the evidence for the third demonstration
#   provider spend, the numerator of the cost reduction figure
#   latency split by cache status
#   a snapshot accessor and a renderer for the exposition format
#
# Why the benchmark reads these same counters: computing a parallel set of
# numbers in the report would allow the service and the report to disagree, and
# the whole project rests on those numbers being trustworthy.
