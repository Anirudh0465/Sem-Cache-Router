# Quality parity scoring with RAGAS.
#
# Will hold:
#   a score record: item, threshold, similarity, answer relevancy, faithfulness
#   a generator for fresh uncached answers, which act as the control
#   a scorer comparing cache hits against those fresh answers at one threshold
#   a plotter for the accuracy against hit rate curve
#
# Why RAGAS rather than an LLM as judge: a judge model would put nondeterminism
# into the one metric the credibility of this project rests on, add a cost that
# would then have to be excluded from the cost figures, and invite the entirely
# fair objection that the judging prompt was tuned until it agreed with the
# desired conclusion. Where RAGAS genuinely cannot express a comparison, a judge
# is used for that narrow case and reported separately.
#
# The curve is the result. A single point on it, chosen after the fact, is not.
