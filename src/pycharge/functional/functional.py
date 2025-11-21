"""This module defines functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c

if TYPE_CHECKING:
    from ..charge import Charge
    from ..types import Scalar, Vector3


def interpolate_position(
    ts: Array,
    position_array: Array,
    velocity_array: Array,
    position_0_fn: Callable[[Scalar], Vector3],
    t_end: None | Array = None,
) -> Callable[[Scalar], Array]:
    t_start = ts[0]
    t_end = ts[-1] if t_end is None else t_end
    t_end_idx = jnp.searchsorted(ts, t_end, side="right") - 1

    def before_start(t: Scalar) -> Array:
        return jnp.asarray(position_0_fn(t), dtype=jnp.result_type(0.0))

    def after_end(t: Scalar) -> Array:
        return position_array[t_end_idx]

    def interpolate(t: Scalar) -> Array:
        t_idx = jnp.searchsorted(ts, t, side="right") - 1

        pos0 = position_array[t_idx]
        vel0 = velocity_array[t_idx]
        pos1 = position_array[t_idx + 1]
        vel1 = velocity_array[t_idx + 1]

        dt = ts[t_idx + 1] - ts[t_idx]
        dpos = pos1 - pos0

        a = (vel0 + vel1) * dt - 2 * dpos
        b = 3 * dpos - (2 * vel0 + vel1) * dt
        c = vel0 * dt
        d = pos0

        t_norm = (t - ts[t_idx]) / dt
        position = a * t_norm**3 + b * t_norm**2 + c * t_norm + d

        return position

    return lambda t: jax.lax.cond(
        t <= t_start, before_start, lambda t: jax.lax.cond(t_end <= t, after_end, interpolate, t), t
    )


def position(t: ArrayLike, charge: Charge) -> Array:
    return jnp.asarray(charge.position_fn(t), dtype=jnp.result_type(0.0))


def velocity(t: ArrayLike, charge: Charge) -> Array:
    return jax.jacobian(position)(t, charge)


def acceleration(t: ArrayLike, charge: Charge) -> Array:
    return jax.jacobian(velocity)(t, charge)


def source_time(r: Array, t: Array, charge: Charge) -> Array:
    """
    Solve for tr such that |r - r_src(tr)| = c * (t - tr).
    """

    def fn_fixed_point(tr, _):
        return t - jnp.linalg.norm(r - position(tr, charge)) / c

    def fn_root_find(tr, _):
        return (t - tr) - jnp.linalg.norm(r - position(tr, charge)) / c

    t_init = t - jnp.linalg.norm(r - position(t, charge)) / c  # Initial guess

    # First use a fixed-point iteration to get close to solution
    solver_fixed_point = optx.FixedPointIteration(rtol=charge.fixed_point_rtol, atol=charge.fixed_point_atol)
    result_fixed_point = optx.fixed_point(
        fn_fixed_point,
        solver_fixed_point,
        t_init,
        max_steps=charge.fixed_point_max_steps,
        throw=charge.fixed_point_throw,
    )
    t_fixed_point = result_fixed_point.value

    # Use Newton's method to refine the solution
    solver_newton = optx.Newton(rtol=charge.root_find_rtol, atol=charge.root_find_atol)
    result = optx.root_find(
        fn_root_find,
        solver_newton,
        t_fixed_point,
        max_steps=charge.root_find_max_steps,
        throw=charge.root_find_throw,
    )
    return result.value
