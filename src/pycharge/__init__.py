"""PyCharge: Differentiable electromagnetics simulation library for moving point charges built on JAX."""

from .charge import Charge
from .potentials_and_fields import potentials_and_fields
from .simulate import simulate
from .sources import Source, dipole_source
from .utils import interpolate_position

__all__ = ["Charge", "potentials_and_fields", "simulate", "Source", "dipole_source", "interpolate_position"]
