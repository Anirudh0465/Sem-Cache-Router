# Workload replay harness.
#
# Will hold:
#   a run configuration record: threshold, caching on or off, workload path,
#   base URL and the git commit, all recorded with the results
#   a result record: request count, cost, p50 and p95 latency, hit ratio per tier
#   a single run function
#   a sweep function that runs caching off, then on at each threshold
#   a writer for structured output
#   a command line entrypoint so the harness runs headless
#
# Threshold is a parameter rather than a constant because the interesting result
# is the curve between hit rate and accuracy, not a single number.
#
# Unfavourable thresholds are run and reported too. A sweep that keeps only the
# flattering number is not a measurement.
