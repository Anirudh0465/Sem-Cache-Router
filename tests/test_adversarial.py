# Adversarial false hit suite.
#
# Will cover:
#   TEST-006  a negated prompt must miss
#   TEST-007  an entity swapped prompt must miss
#   TEST-008  a prompt with one changed number must miss
#   the false hit rate is reported per threshold rather than assumed away
#
# These are the tests expected to be hardest to pass, because they target the
# exact case a semantic cache is designed to get wrong. A failure here is a real
# finding, not a broken test.
