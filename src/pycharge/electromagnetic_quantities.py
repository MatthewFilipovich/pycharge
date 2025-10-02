"""This module defines functions to calculate electromagnetic quantities."""

from typing import Iterable

from jax import Array
from scipy.constants import epsilon_0, mu_0

from pycharge.charge import Charge
from pycharge.config import Config
from pycharge.potentials_and_fields import potentials_and_fields
from pycharge.types import SpaceTimeFn
from pycharge.utils import cross_1d, dot_1d


def _make_fn(charges: Iterable[Charge], quantity: str, config: Config | None) -> SpaceTimeFn:
    fn = potentials_and_fields(charges, **{quantity: True}, config=config)
    return lambda x, y, z, t: fn(x, y, z, t)[quantity]


def scalar_potential(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns scalar potential φ(r, t) from a moving point charge."""
    return _make_fn(charges, "scalar", config)


def vector_potential(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns vector potential A(r, t) from a moving point charge."""
    return _make_fn(charges, "vector", config)


def electric_field(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns electric field E(r, t) from a moving point charge."""
    return _make_fn(charges, "electric", config)


def magnetic_field(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns magnetic field B(r, t) from a moving point charge."""
    return _make_fn(charges, "magnetic", config)


def poynting_vector(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns the Poynting vector S = E × B / μ₀."""
    fn = potentials_and_fields(charges, electric=True, magnetic=True, config=config)

    def poynting_vector_fn(x, y, z, t) -> Array:
        fields = fn(x, y, z, t)
        E = fields["electric"]
        B = fields["magnetic"]
        return cross_1d(E, B) / mu_0

    return poynting_vector_fn


def energy_density(charges: Iterable[Charge], config: Config | None = None) -> SpaceTimeFn:
    """Returns the energy density u = ½ε₀|E|² + ½/μ₀|B|²."""
    fn = potentials_and_fields(charges, electric=True, magnetic=True, config=config)

    def energy_density_fn(x, y, z, t) -> Array:
        fields = fn(x, y, z, t)
        E = fields["electric"]
        B = fields["magnetic"]
        return 0.5 * epsilon_0 * dot_1d(E, E) + 0.5 / mu_0 * dot_1d(B, B)

    return energy_density_fn
