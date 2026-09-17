# OpenAI adapter, the primary provider.
#
# Will hold an OpenAIClient implementing the LLMProvider interface: complete,
# estimate_tokens using tiktoken, and price_of against the committed table.
#
# Credentials are read from the environment and never written into cache
# entries, spans, logs or the benchmark report.
