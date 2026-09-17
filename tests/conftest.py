# Shared fixtures.
#
# Will hold:
#   a fakeredis fixture, so no test needs a running Redis
#   a stub provider that returns canned answers and can be told to fail on
#   demand, with a configurable status code for the breaker tests
#   a settings fixture with small buckets and short TTLs, to keep tests fast
#   a test client fixture wired to the fakes through the real app factory
#
# Why fakes rather than a live stack: the suite has to run with no
# infrastructure, or it will not run often enough to be useful.
#
# Why the client fixture goes through the real app factory: a feature that works
# only when invoked directly from a test is not done.
