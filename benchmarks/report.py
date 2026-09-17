# Benchmark report generation.
#
# Will hold:
#   a reader that pulls counters from the running gateway's /metrics endpoint
#   a builder that renders cost, latency and hit ratio per threshold
#   a writer that stamps the output with the git commit and threshold
#
# Every headline figure is printed next to the accuracy it came at, because a
# cost number without an accuracy number beside it is not a result.
