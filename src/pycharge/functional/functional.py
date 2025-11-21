"""This module defines functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

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

    def interpolate(t) -> Array:
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
