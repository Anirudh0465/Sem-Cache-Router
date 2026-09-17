# Settings, read from the environment.
#
# Will hold a pydantic-settings Settings class with these fields, and no
# constants anywhere else in the codebase:
#
#   redis_url                  where Redis lives
#   chroma_host                where ChromaDB lives
#   similarity_threshold       theta, the Tier 2 acceptance score
#   cache_ttl_seconds          entry lifetime, applied at write time
#   bucket_capacity            maximum tokens a caller may hold
#   bucket_refill_rate         tokens restored per second
#   primary_provider           the provider tried first
#   secondary_provider         the failover target
#   breaker_failure_threshold  failures needed to trip the breaker open
#   breaker_window_seconds     the window those failures are counted in
#   breaker_base_backoff       starting retry delay
#   breaker_backoff_ceiling    the cap that exponential growth stops at
#   embedding_model            which sentence-transformers model to load
#   log_prompts                whether prompt text reaches the logs
#
# Also a cached accessor so the environment is read once per process.
#
# Why every value lives here: a benchmark has to sweep thresholds, TTLs and
# bucket sizes without a code change, or a run is not reproducible from a
# committed configuration.
