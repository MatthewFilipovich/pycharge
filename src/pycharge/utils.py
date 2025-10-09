"""This module defines utility functions."""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def interpolate_position(
    ts: Array,
    position_array: Array,
    velocity_array: Array,
    position_0: Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]],
    nan_position_fill_value: Array | None = None,
) -> Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]]:
    t_start = ts[0]
    t_end = ts[-1]
    if nan_position_fill_value is None:
        nan_position_fill_value = jnp.full(3, jnp.nan)

    def before_start(t):
        return position_0(t)

    def after_end(t):
        return nan_position_fill_value

    def interpolate(t):
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

        position = jnp.where(jnp.any(jnp.isnan(position)), nan_position_fill_value, position)

        return position

    return lambda t: jax.lax.cond(
        t < t_start, before_start, lambda t: jax.lax.cond(t_end < t, after_end, interpolate, t), t
    )
