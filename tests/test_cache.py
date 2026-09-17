# Cache tier tests.
#
# Will cover:
#   TEST-001  an identical prompt sent twice returns exact_hit, no provider call
#   TEST-002  the same prompt against a different model is a separate entry
#   TEST-003  a whitespace only difference hashes the same
#   a case difference does not hit Tier 1, since case is preserved on purpose
#   TEST-004  a paraphrase above theta returns semantic_hit with its similarity
#   TEST-005  a paraphrase below theta misses and the provider is called
#   TEST-009  an entry past its TTL is not served and is removed from both tiers
#   a hit preserves the remaining TTL rather than extending it
#   TEST-015  with Chroma unreachable, Tier 1 keeps serving and the degradation
#             is recorded, rather than the request failing
