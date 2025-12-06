"""PyCharge: Electromagnetics simulation library for moving point charges built on JAX."""

import pycharge.functional as functional
from pycharge.charge import Charge, SolverConfig
from pycharge.potentials_and_fields import potentials_and_fields
from pycharge.simulate import simulate
from pycharge.sources import Source, dipole_source, free_particle_source

__all__ = [
    "Charge",
    "dipole_source",
    "free_particle_source",
    "functional",
    "potentials_and_fields",
    "simulate",
    "SolverConfig",
    "Source",
]
