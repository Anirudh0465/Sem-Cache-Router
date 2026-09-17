# Local sentence-transformers encoder.
#
# Will hold an Embedder class with:
#   load       pull the model weights from the image into memory at startup
#   encode     embed one prompt, running the model in a thread pool executor
#   dimension  the embedding width, needed when creating the Chroma collection
#
# Why encode never runs on the event loop: the call is CPU bound, and awaiting
# it directly would stall every other in flight request for the duration of a
# Tier 2 lookup. That would surface as a latency regression under concurrency
# and would be easy to misdiagnose as slow vector search.
#
# Why the model is local rather than a hosted API: a paid embedding call on
# every Tier 2 lookup, including the ones that miss, would have to be netted out
# of the headline cost figure or it would silently inflate it. It would also add
# a second external dependency to the request path of a project whose other
# claim is about surviving external failures.
