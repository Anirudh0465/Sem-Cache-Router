# Load profile for the concurrency target.
#
# Will hold a Locust user that mixes three task weights:
#   repeat prompts, exercising the Tier 1 path
#   paraphrases, exercising embedding and vector search
#   fresh prompts, exercising the full provider path
#
# Verifies the throughput claim of 50 concurrent requests on a single container
# with no queue growth, and checks that p95 latency stays inside target. That is
# an assertion, so it needs a measurement behind it.
