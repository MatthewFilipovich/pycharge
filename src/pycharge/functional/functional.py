"""Core utility functions used in PyCharge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c

if TYPE_CHECKING:
    from pycharge.charge import Charge
    from pycharge.types import Scalar, Vector3


def interpolate_position(
    ts: Array,
    position_array: Array,
    velocity_array: Array,
    position_0_fn: Callable[[Scalar], Vector3],
    t_end: None | Array = None,
) -> Callable[[Scalar], Array]:
    r"""Create position interpolation function from trajectory data.

    Uses cubic Hermite interpolation for :math:`C^1` continuity (continuous position and velocity).
    The interpolated position :math:`\mathbf{r}(t)` for :math:`t_i \leq t \leq t_{i+1}` is:

    .. math::

        \mathbf{r}(t) = a\tau^3 + b\tau^2 + c\tau + d

    where :math:`\tau = (t - t_i)/(t_{i+1} - t_i)` and coefficients ensure matching positions
    and velocities at interval endpoints.

    Args:
        ts (Array): Time points, shape ``(n_steps,)``.
        position_array (Array): Positions at each time, shape ``(n_steps, 3)``.
        velocity_array (Array): Velocities at each time, shape ``(n_steps, 3)``.
        position_0_fn (Callable[[Scalar], Vector3]): Original position function before simulation start.
        t_end (Array or None): End time for interpolation. Default: if ``None``, uses ``ts[-1]``.

    Returns:
        Callable[[Scalar], Array]: Function returning position at time t.
            For :math:`t \leq t_0`, returns ``position_0_fn(t)``. For :math:`t \geq t_{\mathrm{end}}`,
            returns final position. Otherwise, cubic Hermite interpolation.
    """
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
    r"""Compute charge position at time t.

    Args:
        t (ArrayLike): Time (scalar or array).
        charge (Charge): Charge object.

    Returns:
        Array: Position :math:`\mathbf{r}(t) = [x, y, z]`.
    """
    return jnp.asarray(charge.position_fn(t), dtype=jnp.result_type(0.0))


def velocity(t: ArrayLike, charge: Charge) -> Array:
    r"""Compute charge velocity :math:`\mathbf{v}(t) = d\mathbf{r}/dt` via automatic differentiation.

    Args:
        t (ArrayLike): Time (scalar or array).
        charge (Charge): Charge object.

    Returns:
        Array: Velocity :math:`\mathbf{v}(t) = [v_x, v_y, v_z]`.
    """
    return jax.jacobian(position)(t, charge)


def acceleration(t: ArrayLike, charge: Charge) -> Array:
    r"""Compute charge acceleration :math:`\mathbf{a}(t) = d^2\mathbf{r}/dt^2` via automatic differentiation.

    Args:
        t (ArrayLike): Time (scalar or array).
        charge (Charge): Charge object.

    Returns:
        Array: Acceleration :math:`\mathbf{a}(t) = [a_x, a_y, a_z]`.
    """
    return jax.jacobian(velocity)(t, charge)


def emission_time(r: Array, t: Array, charge: Charge) -> Array:
    r"""Compute emission time (retarded time) :math:`t_r` for a charge at observation point.

    Solves :math:`t_r = t - \frac{1}{c}\,|\mathbf{r} - \mathbf{r}_s(t_r)|` using fixed-point iteration
    followed by Newton refinement.

    Args:
        r (Array): Observation point :math:`\mathbf{r} = [x, y, z]`.
        t (Array): Observation time.
        charge (Charge): Charge with position function and solver config.

    Returns:
        Array: :math:`t_r`.

    Note:
        Solver parameters (``rtol``, ``atol``, ``max_steps``, ``throw``) configured via ``charge.solver_config``.
    """
    config = charge.solver_config

    def fn_fixed_point(tr, _):
        return t - jnp.linalg.norm(r - position(tr, charge)) / c

    def fn_root_find(tr, _):
        return (t - tr) - jnp.linalg.norm(r - position(tr, charge)) / c

    t_init = t - jnp.linalg.norm(r - position(t, charge)) / c  # Initial guess

    # First use a fixed-point iteration to get close to solution
    solver_fixed_point = optx.FixedPointIteration(rtol=config.fixed_point_rtol, atol=config.fixed_point_atol)
    result_fixed_point = optx.fixed_point(
        fn_fixed_point,
        solver_fixed_point,
        t_init,
        max_steps=config.fixed_point_max_steps,
        throw=config.fixed_point_throw,
    )
    t_fixed_point = result_fixed_point.value

    # Use Newton's method to refine the solution
    solver_newton = optx.Newton(rtol=config.root_find_rtol, atol=config.root_find_atol)
    result = optx.root_find(
        fn_root_find,
        solver_newton,
        t_fixed_point,
        max_steps=config.root_find_max_steps,
        throw=config.root_find_throw,
    )
    return result.value
