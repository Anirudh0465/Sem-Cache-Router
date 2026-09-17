# Prompt normalisation and Tier 1 key derivation.
#
# Will hold:
#   a schema version constant and the Redis key prefix
#   normalise_messages  collapse runs of whitespace and trim each message
#   make_key            SHA-256 over the model and the normalised messages,
#                       prefixed with the schema version
#
# Normalisation folds whitespace and does nothing else. Case and punctuation are
# preserved on purpose: two prompts differing only in case are not reliably the
# same question, and folding them here would hide a correctness decision inside
# what looks like a performance optimisation. That kind of near match belongs in
# Tier 2, where the threshold makes the risk explicit and measurable.
#
# Why the model is part of the key: the same prompt against a different model is
# a different answer.
#
# Why the key carries a schema version: if the stored entry shape changes, old
# entries are orphaned and expire on their existing TTL, instead of needing a
# migration or a flush that would invalidate a benchmark baseline.
