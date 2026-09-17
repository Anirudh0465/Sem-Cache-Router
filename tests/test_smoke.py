# Import every module.
#
# Will hold one parametrised test that imports each module in the project.
#
# Exists so CI is green from the first commit, and so a broken import or an
# accidental import time side effect is caught by the test suite rather than at
# container start.
