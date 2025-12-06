"""Core functional utilities for charge dynamics and retarded time calculations.

This module provides essential functions for computing charge positions, velocities,
accelerations, and retarded times needed for electromagnetic field calculations using
the Liénard-Wiechert potentials.
"""

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
    """Create a position interpolation function from simulation trajectory data.

    Constructs a cubic Hermite interpolation function for charge position using position
    and velocity data from a time-stepped simulation. Outside the simulation time range,
    returns the original position function or the final position.

    Parameters
    ----------
    ts : Array
        1D array of simulation time points, shape ``(n_steps,)``.
    position_array : Array
        2D array of position vectors at each time step, shape ``(n_steps, 3)``.
    velocity_array : Array
        2D array of velocity vectors at each time step, shape ``(n_steps, 3)``.
    position_0_fn : Callable[[Scalar], Vector3]
        Original position function used before simulation start time.
    t_end : Array or None, optional
        Optional end time for interpolation. If None, uses ``ts[-1]``.

    Returns
    -------
    Callable[[Scalar], Array]
        A callable that takes time t and returns the interpolated 3D position vector:

        - For t <= ts[0]: returns position_0_fn(t)
        - For t >= t_end: returns position_array[t_end_idx]
        - Otherwise: returns cubic Hermite interpolation

    Notes
    -----
    Uses cubic Hermite interpolation which ensures C1 continuity (continuous
    position and velocity) across time step boundaries.
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
    r"""Compute the position of a charge at a given time.

    Parameters
    ----------
    t : ArrayLike
        Time value (scalar or array).
    charge : Charge
        Charge object with position_fn attribute.

    Returns
    -------
    Array
        3D position vector :math:`\mathbf{r}(t) = [x(t), y(t), z(t)]` as a JAX array.
    """
    return jnp.asarray(charge.position_fn(t), dtype=jnp.result_type(0.0))


def velocity(t: ArrayLike, charge: Charge) -> Array:
    r"""Compute the velocity of a charge at a given time via automatic differentiation.

    Calculates :math:`\mathbf{v}(t) = d\mathbf{r}/dt` using JAX's automatic differentiation.

    Parameters
    ----------
    t : ArrayLike
        Time value (scalar or array).
    charge : Charge
        Charge object with position_fn attribute.

    Returns
    -------
    Array
        3D velocity vector :math:`\mathbf{v}(t) = [v_x(t), v_y(t), v_z(t)]` as a JAX array.

    Notes
    -----
    Requires the position function to be differentiable. Uses JAX's jacobian
    for automatic differentiation.
    """
    return jax.jacobian(position)(t, charge)


def acceleration(t: ArrayLike, charge: Charge) -> Array:
    r"""Compute the acceleration of a charge at a given time via automatic differentiation.

    Calculates :math:`\mathbf{a}(t) = d^2\mathbf{r}/dt^2` using JAX's automatic differentiation.

    Parameters
    ----------
    t : ArrayLike
        Time value (scalar or array).
    charge : Charge
        Charge object with position_fn attribute.

    Returns
    -------
    Array
        3D acceleration vector :math:`\mathbf{a}(t) = [a_x(t), a_y(t), a_z(t)]` as a JAX array.

    Notes
    -----
    Requires the position function to be twice differentiable. Computed by
    differentiating the velocity function.
    """
    return jax.jacobian(velocity)(t, charge)


def source_time(r: Array, t: Array, charge: Charge) -> Array:
    r"""Compute the retarded time for electromagnetic field calculations.

    Solves the retarded time equation:

    .. math::

        t_{\text{ret}} = t - \frac{|\mathbf{r} - \mathbf{r}_{\text{src}}(t_{\text{ret}})|}{c}

    This is the time at which a signal must have been emitted from the source charge to
    reach the observation point :math:`\mathbf{r}` at time :math:`t`, accounting for
    light-speed propagation.

    The solution uses a two-stage approach:

    1. Fixed-point iteration to get close to the solution
    2. Newton's method refinement for high accuracy

    Parameters
    ----------
    r : Array
        3D observation point position vector :math:`\mathbf{r} = [x, y, z]`.
    t : Array
        Observation time :math:`t`.
    charge : Charge
        Charge object containing position function and solver parameters.

    Returns
    -------
    Array
        Retarded time :math:`t_{\text{ret}}` as a JAX scalar array.

    Notes
    -----
    The solver parameters (rtol, atol, max_steps, throw) can be configured in
    the Charge object to control convergence behavior and accuracy.

    References
    ----------
    .. [1] Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.). Wiley.
           Section 14.1: Liénard-Wiechert Potentials.
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
