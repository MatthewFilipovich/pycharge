"""This module defines functions to calculate electromagnetic quantities."""
from typing import Callable, Iterable, Literal

from jax import Array
from jax.typing import ArrayLike
from scipy.constants import epsilon_0, mu_0
from typing_extensions import TypeAlias

from .charge import Charge
from .potentials_and_fields import potentials_and_fields
from .utils import cross_1d, dot_1d

SpaceTimeFn: TypeAlias = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]
FieldComponent: TypeAlias = Literal["total", "velocity", "acceleration"]


def _make_fn(
    charges: Iterable[Charge],
    field_name: str,
    field_component: FieldComponent = "total",
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    fn = potentials_and_fields(
        charges,
        **{field_name: True},
        field_component=field_component,
        solver_config=solver_config,
    )
    return lambda x, y, z, t: fn(x, y, z, t)[field_name]


def scalar_potential(charges: Iterable[Charge], solver_config: dict | None = None) -> SpaceTimeFn:
    """Returns scalar potential φ(r, t) from a moving point charge."""
    return _make_fn(charges, "scalar", solver_config=solver_config)


def vector_potential(
    charges: Iterable[Charge],
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    """Returns vector potential A(r, t) from a moving point charge."""
    return _make_fn(charges, "vector", solver_config=solver_config)


def electric_field(
    charges: Iterable[Charge],
    field_component: FieldComponent = "total",
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    """Returns electric field E(r, t) from a moving point charge."""
    return _make_fn(charges, "electric", field_component=field_component, solver_config=solver_config)


def magnetic_field(
    charges: Iterable[Charge],
    field_component: FieldComponent = "total",
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    """Returns magnetic field B(r, t) from a moving point charge."""
    return _make_fn(charges, "magnetic", field_component=field_component, solver_config=solver_config)


def poynting_vector(
    charges: Iterable[Charge],
    field_component: FieldComponent = "total",
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    """Returns the Poynting vector S = E × B / μ₀."""
    fn = potentials_and_fields(
        charges, electric=True, magnetic=True, field_component=field_component, solver_config=solver_config
    )

    def poynting_vector_fn(x, y, z, t) -> Array:
        fields = fn(x, y, z, t)
        E = fields["electric"]
        B = fields["magnetic"]
        return cross_1d(E, B) / mu_0

    return poynting_vector_fn


def energy_density(
    charges: Iterable[Charge],
    field_component: FieldComponent = "total",
    solver_config: dict | None = None,
) -> SpaceTimeFn:
    """Returns the energy density u = ½ε₀|E|² + ½/μ₀|B|²."""
    fn = potentials_and_fields(
        charges,
        electric=True,
        magnetic=True,
        field_component=field_component,
        solver_config=solver_config,
    )

    def energy_density_fn(x, y, z, t) -> Array:
        fields = fn(x, y, z, t)
        E = fields["electric"]
        B = fields["magnetic"]
        return 0.5 * epsilon_0 * dot_1d(E, E) + 0.5 / mu_0 * dot_1d(B, B)

    return energy_density_fn
