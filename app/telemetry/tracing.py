# OpenTelemetry span setup.
#
# Will hold:
#   the gen_ai and semcache attribute name constants
#   a function to install the tracer provider and exporter
#   a context manager that opens a span for one request
#   a helper that attaches the cache outcome to a span
#
# Attributes carried on every request span:
#   gen_ai.system, gen_ai.request.model
#   gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
#   semcache.cache_status, semcache.similarity_score
#   semcache.breaker_state, semcache.cost_usd
#
# Why tracing rather than logging alone: the interesting questions here are per
# request and comparative. Which requests hit, at what similarity, at what cost,
# in which breaker state. A log line answers that for one request; a trace
# answers it for a distribution.
#
# Prompt text is recorded only in a development configuration, since prompts are
# caller data.
