"""This module defines the function for calculating the potentials and fields."""
from typing import Callable, Iterable, Literal

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c, epsilon_0, pi

from .charge import Charge


def potentials_and_fields(
    charges: Iterable[Charge],
    *,
    scalar: bool = False,
    vector: bool = False,
    electric: bool = False,
    magnetic: bool = False,
    field_component: Literal["total", "velocity", "acceleration"] = "total",
    solver_config: dict | None = None,
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], dict[str, Array]]:
    """
    Returns a function to compute electromagnetic fields and potentials
    from moving point charges at given spacetime points.

    Args:
        charges: An iterable containing Charge objects.
        scalar: If True, compute scalar potential φ.
        vector: If True, compute vector potential A.
        electric: If True, compute electric field E.
        magnetic: If True, compute magnetic field B.
        field_component: Specifies the component of the electric and magnetic field to compute:
            'total', 'velocity', or 'acceleration'. Default is 'total'.
        solver_config: A dictionary of keyword arguments passed to the root finder.

    Returns:
        A function that takes (x, y, z, t) arrays and returns a dictionary of quantities.
    """

    # Ensure charges is an iterable
    if isinstance(charges, Charge):
        charges = [charges]

    if field_component not in {"total", "velocity", "acceleration"}:
        raise ValueError(
            f"Invalid field_component: {field_component}. Must be 'total', 'velocity', or 'acceleration'."
        )

    solver_config = solver_config or {}

    # Precompute functions for each charge
    # TODO: replace for loop with jax.lax.scan?
    position_fns = [lambda t, c=charge: jnp.asarray(c.position(t)) for charge in charges]  # Convert to array
    velocity_fns = [jax.jacobian(pos_fn) for pos_fn in position_fns]
    acceleration_fns = [jax.jacobian(vel_fn) for vel_fn in velocity_fns]
    source_time_fns = [source_time(charge, **solver_config) for charge in charges]
    qs = jnp.array([charge.q for charge in charges])

    def potentials_and_fields_fn(x: ArrayLike, y: ArrayLike, z: ArrayLike, t: ArrayLike) -> dict[str, Array]:
        """
        Compute scalar and vector potentials, and electric and magnetic fields at
        spatial coordinates (x, y, z) and time t.

        Shapes:
            x, y, z, t: same shape, e.g., (...,)
            Returns: fields reshaped to match input shape, with vector fields having final dim 3.
        """
        x, y, z, t = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(t)
        if not (x.shape == y.shape == z.shape == t.shape):
            raise ValueError("x, y, z, and t must have the same shape.")

        r = jnp.stack([x, y, z], axis=-1)  # Stack into (..., 3)
        r_flat, t_flat = r.reshape(-1, 3), t.ravel()  # Flatten for vmap

        quantities_flat = jax.vmap(calculate_total_sources)(r_flat, t_flat)
        return {  # Reshape back to original shape
            key: value.reshape(x.shape) if key == "scalar" else value.reshape(*x.shape, 3)
            for key, value in quantities_flat.items()
        }

    def calculate_total_sources(r: Array, t: Array) -> dict[str, Array]:
        """
        Computes the total fields at a single point (r, t) by summing contributions from all charges.
        """

        def calculate_individual_source(
            r_src: Array, v_src: Array, a_src: Array, q: Array
        ) -> dict[str, Array]:
            """
            Computes the fields from a single charge at a single point (r, t).
            """

            R = jnp.linalg.norm(r - r_src)  # Distance from source to observation point
            n = (r - r_src) / R  # Unit vector from source to observation
            beta = v_src / c  # Velocity normalized to speed of light
            beta_dot = a_src / c  # Acceleration normalized
            one_minus_n_dot_beta = 1 - jnp.dot(n, beta)  # (1 - n · β)
            coeff = q / (4 * pi * epsilon_0)  # Common prefactor
            quantities = {}

            # Potentials
            if scalar or vector:
                phi = coeff / (one_minus_n_dot_beta * R)

                if scalar:
                    quantities["scalar"] = phi
                if vector:
                    quantities["vector"] = beta * phi / c

            # Fields
            if electric or magnetic:
                gamma_sq_inv = 1 - jnp.dot(beta, beta)  # 1/γ²
                one_minus_n_dot_beta_cubed = one_minus_n_dot_beta**3
                n_minus_beta = n - beta

                E = 0
                if field_component in ("total", "velocity"):  # Electric field from velocity term
                    E += n_minus_beta / (gamma_sq_inv * one_minus_n_dot_beta_cubed * R**2)
                if field_component in ("total", "acceleration"):  # Electric field from acceleration term
                    E += jnp.cross(n, jnp.cross(n_minus_beta, beta_dot)) / (
                        c * one_minus_n_dot_beta_cubed * R
                    )
                E *= coeff

                if electric:
                    quantities["electric"] = E
                if magnetic:
                    quantities["magnetic"] = jnp.cross(n, E) / c

            return quantities

        # Solve for retarded times for each charge
        t_srcs = jnp.stack([fn(r, t) for fn in source_time_fns])
        # Evaluate source properties at the retarded times
        r_srcs = jnp.stack([pos_fn(tr) for pos_fn, tr in zip(position_fns, t_srcs)])
        v_srcs = jnp.stack([vel_fn(tr) for vel_fn, tr in zip(velocity_fns, t_srcs)])
        a_srcs = jnp.stack([acc_fn(tr) for acc_fn, tr in zip(acceleration_fns, t_srcs)])

        # Compute individual contributions
        individual_quantities = jax.vmap(calculate_individual_source)(r_srcs, v_srcs, a_srcs, qs)
        # Sum contributions from all charges
        return {key: jnp.sum(value, axis=0) for key, value in individual_quantities.items()}

    return potentials_and_fields_fn


def source_time(
    charge: Charge, rtol: float = 0, atol: float = 1.48e-8, max_steps: int = 256
) -> Callable[[Array, Array], Array]:
    """
    Returns a function to compute the retarded time for a given field point and observation time.

    Args:
        charge: Charge object containing the trajectory.
        rtol: Relative tolerance for the solver.
        atol: Absolute tolerance for the solver.
        max_steps: Maximum number of solver iterations.

    Returns:
        Function that takes (r, t) and returns the retarded time tr.
    """

    def source_time_fn(r: Array, t: Array) -> Array:
        """
        Solve for tr such that |r - r_src(tr)| = c * (t - tr).
        """

        def fn(tr, args):
            return jnp.linalg.norm(r - jnp.asarray(charge.position(tr))) - c * (t - tr)

        solver = optx.Newton(rtol, atol)
        t_init = t - jnp.linalg.norm(r - jnp.asarray(charge.position(t))) / c  # Initial guess
        result = optx.root_find(fn, solver, t_init, max_steps=max_steps)
        return result.value

    return source_time_fn
