# Workload loading.
#
# Will hold:
#   a workload item record: id, prompt, and the paraphrase group it belongs to
#   a loader that reads the committed subset from disk
#   a sampler that draws the subset once, for committing
#
# The reference workload is a fixed subset of SemBenchmarkLmArena, chosen
# because it contains genuine paraphrase pairs, which is exactly the traffic a
# semantic cache is meant to exploit, and because it is the dataset used by the
# published benchmark this project takes as its reference point.
#
# The subset is sampled once and committed. Re sampling between runs would make
# two results incomparable, and loading from disk rather than fetching means a
# run needs no network and cannot silently change.
