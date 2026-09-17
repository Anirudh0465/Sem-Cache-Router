# Wire and internal schemas.
#
# Will hold pydantic models:
#
#   CacheStatus             exact_hit, semantic_hit or miss
#   Message                 one chat message, role and content
#   ChatCompletionRequest   model, messages, temperature, max_tokens, stream
#   Usage                   prompt, completion and total token counts
#   Choice                  one completion choice
#   ChatCompletionResponse  the OpenAI response shape plus two additive fields,
#                           semcache_status and semcache_similarity
#   ErrorResponse           code, message, retry_after_seconds, request_id
#   CacheEntry              one logical cached response, written to both tiers
#   Reservation             a held token estimate, awaiting reconciliation
#   ProviderResponse        a provider reply normalised across vendors
#
# Why the OpenAI shape: the gateway earns nothing by inventing its own
# contract, and matching the existing one means a caller changes a base URL and
# nothing else. The two extra fields are additive, so a strict client that
# ignores unknown keys still parses the body.
