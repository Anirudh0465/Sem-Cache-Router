# Anthropic adapter, the failover target.
#
# Will hold an AnthropicClient implementing the same LLMProvider interface as
# the primary: complete, estimate_tokens and price_of.
#
# Anthropic reports usage in a different shape from OpenAI. Absorbing that
# difference here is the entire point of the adapter, because it leaves cost
# accounting with one code path rather than one per vendor.
