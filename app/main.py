# Application entrypoint.
#
# Will hold:
#   a lifespan context manager that builds redis, chroma, the embedder, the
#   provider adapters and the breakers, and attaches them to app.state
#   an app factory that registers the chat and health routers
#
# Why wiring happens in lifespan rather than at import time: nothing should
# connect to an external service just because a module was imported, and the
# test suite needs to substitute a fake Redis and a stub provider without
# patching module globals.
