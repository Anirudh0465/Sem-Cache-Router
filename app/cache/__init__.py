# cache package
#
# The two tier cache: an exact hash tier in front of a semantic tier.
#
# Why two tiers rather than one: embedding is the most expensive step in the
# lookup path, and literal repeats are common and do not need it. Tier 1 catches
# those in about a millisecond, and only a Tier 1 miss pays for embedding and
# vector search.
