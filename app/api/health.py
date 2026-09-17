# Liveness and metrics routes.
#
# Will hold:
#   GET /health    liveness plus whether redis and chroma are reachable
#   GET /metrics   aggregate counters in the format the benchmark report reads
#
# Why dependency state is on /health: losing chroma degrades the gateway while
# losing redis stops it, and an operator has to be able to tell those apart.
#
# Why /metrics feeds the report directly: the harness reads the same numbers the
# service reports, so the two cannot quietly disagree about what happened.
