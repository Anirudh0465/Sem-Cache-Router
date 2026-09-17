# The provider interface.
#
# Will hold an LLMProvider abstract base class with:
#   complete          issue a completion, returning the normalised shape
#   estimate_tokens   estimate prompt cost before the call is made
#   price_of          convert reported usage to dollars
#
# Why a common interface: it makes adding a second provider a configuration
# change rather than new code, and it keeps the breaker from being coupled to
# one vendor SDK.
#
# Why price comes from a committed table rather than a constant: a provider
# price change would otherwise silently invalidate every recorded benchmark
# result.
