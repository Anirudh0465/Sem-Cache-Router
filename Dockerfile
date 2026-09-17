# Gateway image.
#
# Will do, in order:
#   start from a slim Python base
#   install dependencies from requirements.txt as its own layer, so a code
#   change does not reinstall the world
#   download the sentence-transformers weights at build time
#   copy the application source
#   run as a non root user
#   start the ASGI server
#
# Why the model weights are baked in at build time rather than fetched on first
# use: a running container then needs no outbound network access to serve a
# request, which matters both for a locked down environment and for a project
# whose headline claim is about surviving external failures.
#
# No secrets are baked into the image. Provider keys arrive through the
# environment at run time.
