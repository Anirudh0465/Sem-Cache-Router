# Tier 2: semantic cache backed by ChromaDB.
#
# Will hold a SemanticCache class with:
#   search          nearest entry and its score, if the score meets theta
#   add             index a response against its prompt embedding
#   evict_expired   remove entries past their TTL, since Chroma has no native one
#   is_available    whether Chroma is reachable
#
# This tier is the actual subject of the project. Unlike a normal cache, a hit
# here can be wrong, so every hit carries the similarity score that produced it
# and that score is recorded on the trace span. That is what makes a wrong
# answer traceable back to the decision that produced it.
#
# is_available returning false makes the gateway serve Tier 1 only and record
# the degradation, rather than failing the request. Losing the vector store
# should cost hit rate, not availability.
