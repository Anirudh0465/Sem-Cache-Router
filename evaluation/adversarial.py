# Adversarial false hit generators.
#
# Will hold:
#   a case record: the original prompt, the mutation, and the expectation
#   a negation generator
#   an entity swap generator
#   a numeric change generator
#   a suite builder across all three mutation kinds
#   a false hit rate calculator
#
# These target the cases where two prompts sit close together in embedding space
# while meaning different things, which is the failure mode a semantic cache is
# built to commit. They are expected to be the hardest tests to pass.
#
# The false hit rate is reported as a headline metric rather than buried, since
# it is the honest cost of the cost saving.
