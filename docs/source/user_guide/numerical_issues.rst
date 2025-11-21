Numerical Issues
================

Discuss how we can't accurately represent 1 + 1e-20... important for time.

OFten recommended to import with float64 for higher precision calculations.
jax.config.update("jax_enable_x64", True)
This can be done at the top of the main script or in the test files where higher precision is needed.