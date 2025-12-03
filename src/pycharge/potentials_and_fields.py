from typing import Callable, Iterable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c, epsilon_0, pi

from pycharge.charge import Charge
from pycharge.functional import acceleration, position, source_time, velocity


class Quantities(NamedTuple):
    "Quantities!"

    scalar: Array
    vector: Array
    electric: Array
    magnetic: Array

    electric_term1: Array
    electric_term2: Array
    magnetic_term1: Array
    magnetic_term2: Array


def potentials_and_fields(
    charges: Iterable[Charge],
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Quantities]:
    "potentials_and_fields!"
    charges = [charges] if isinstance(charges, Charge) else list(charges)

    if len(charges) == 0:
        raise ValueError("At least one charge must be provided.")

    qs = jnp.array([charge.q for charge in charges])

    def quantities_fn(x: ArrayLike, y: ArrayLike, z: ArrayLike, t: ArrayLike) -> Quantities:
        x, y, z, t = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(t)
        if not (x.shape == y.shape == z.shape == t.shape):
            raise ValueError("x, y, z, and t must have the same shape.")
        original_shape = x.shape

        r = jnp.stack([x, y, z], axis=-1)  # Stack into (..., 3)
        r_flat, t_flat = r.reshape(-1, 3), t.ravel()  # Flatten for vmap

        quantities_flat = jax.vmap(calculate_total_sources)(r_flat, t_flat)
        scalar_flat, other_quantities_flat = quantities_flat[0], quantities_flat[1:]

        return Quantities(
            scalar_flat.reshape(original_shape),  # 1D scalar quantity
            *(q.reshape(*original_shape, 3) for q in other_quantities_flat),  # 3D vectors quantities
        )

    def calculate_total_sources(r: Array, t: Array) -> Quantities:
        # Solve for retarded times for each charge
        t_srcs = jnp.stack([source_time(r, t, charge) for charge in charges])
        # Evaluate source properties at the retarded times
        r_srcs = jnp.stack([position(tr, charge) for tr, charge in zip(t_srcs, charges)])
        v_srcs = jnp.stack([velocity(tr, charge) for tr, charge in zip(t_srcs, charges)])
        a_srcs = jnp.stack([acceleration(tr, charge) for tr, charge in zip(t_srcs, charges)])

        # Compute individual contributions
        calculate_individual_source_vmap = jax.vmap(calculate_individual_source, in_axes=(0, 0, 0, 0, None))
        individual_quantities = calculate_individual_source_vmap(r_srcs, v_srcs, a_srcs, qs, r)
        # Sum contributions from all charges
        summed_quantities = Quantities(*(jnp.sum(value, axis=0) for value in individual_quantities))

        return summed_quantities

    def calculate_individual_source(
        r_src: Array, v_src: Array, a_src: Array, q: Array, r: Array
    ) -> Quantities:
        R = jnp.linalg.norm(r - r_src)  # Distance from source to observation point
        n = (r - r_src) / R  # Unit vector from source to observation
        β = v_src / c  # Velocity normalized to speed of light
        β̇ = a_src / c  # Acceleration normalized to speed of light
        one_minus_β_dot_β = 1 - jnp.dot(β, β)
        one_minus_n_dot_β = 1 - jnp.dot(n, β)
        one_minus_n_dot_β_cubed = one_minus_n_dot_β**3
        n_minus_β = n - β
        coeff = q / (4 * pi * epsilon_0)  # Common coefficient

        # Potentials
        φ = coeff / (one_minus_n_dot_β * R)
        A = β * φ / c

        # Fields
        E1 = coeff * n_minus_β * one_minus_β_dot_β / (one_minus_n_dot_β_cubed * R**2)
        E2 = coeff * jnp.cross(n, jnp.cross(n_minus_β, β̇)) / (c * one_minus_n_dot_β_cubed * R)
        E = E1 + E2

        B1 = jnp.cross(n, E1) / c
        B2 = jnp.cross(n, E2) / c
        B = B1 + B2

        return Quantities(φ, A, E, B, E1, E2, B1, B2)

    return quantities_fn
