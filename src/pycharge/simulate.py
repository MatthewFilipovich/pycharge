"""Time-stepping simulation of interacting charged particle systems.

This module provides functions for simulating the dynamics of multiple interacting
charged particles using ordinary differential equation (ODE) solvers with
electromagnetic force coupling.
"""

from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from pycharge.functional import interpolate_position, position, velocity
from pycharge.sources import Source


def simulate(sources: Sequence[Source], ts: Array, print_every: int = 100) -> Callable[[], tuple[Array, ...]]:
    """Create a simulation function for time-evolution of charged particle sources.

    Constructs a function that simulates the dynamics of multiple interacting charged
    particle sources over specified time steps. Each source evolves according to its
    ODE function while interacting electromagnetically with charges from other sources.

    The simulation uses RK4 (4th-order Runge-Kutta) time-stepping and maintains trajectory
    data for position and velocity interpolation.

    Parameters
    ----------
    sources : Sequence[Source]
        Sequence of Source objects to simulate.
    ts : Array
        1D array of time points at which to compute the state, shape ``(n_steps,)``.
        Must be uniformly or non-uniformly spaced.
    print_every : int, optional
        Print progress every N time steps. Set to 0 or None to disable
        progress printing. Default is 100.

    Returns
    -------
    Callable[[], tuple[Array, ...]]
        A callable ``simulate_fn()`` that when called, performs the simulation and returns
        a tuple of state arrays, one per source. Each state array has shape
        ``(n_steps, n_charges, 2, 3)`` containing ``[position, velocity]`` for each charge
        at each time step.

    Notes
    -----
    - Uses JAX's lax.fori_loop for efficient compilation
    - Updates charge position functions at each step for self-consistent field calculations
    - Compatible with JAX JIT compilation for performance
    - Memory scales as O(n_steps * n_sources * n_charges * 6)

    See Also
    --------
    Source : Container for charge system and ODE function
    dipole_source : Factory function for dipole oscillators
    free_particle_source : Factory function for free particles
    """
    dts = ts[1:] - ts[:-1]

    def simulate_fn():
        initial_source_states = tuple(create_initial_state(source) for source in sources)
        final_source_states = jax.lax.fori_loop(0, len(ts) - 1, time_step_body, initial_source_states)

        return final_source_states

    def create_initial_state(source: Source) -> Array:
        source_state = jnp.full([len(ts), len(source.charges_0), 2, 3], jnp.nan)
        for charge_idx, charge in enumerate(source.charges_0):
            pos0, vel0 = position(ts[0], charge), velocity(ts[0], charge)
            source_state = source_state.at[0, charge_idx, :, :].set([pos0, vel0])

        return source_state

    def time_step_body(time_idx: int, source_states: tuple[Array, ...]) -> tuple[Array, ...]:
        print_timestep(time_idx)
        charges = create_charges(time_idx, source_states)

        def time_step_source(source_idx):
            ode_func = sources[source_idx].ode_func
            t = ts[time_idx]
            u = source_states[source_idx][time_idx]
            dt = dts[time_idx]
            other_charges_flat = [c for i, c_tuple in enumerate(charges) if i != source_idx for c in c_tuple]
            u_step = rk4_step(ode_func, t, u, dt, other_charges_flat)

            return source_states[source_idx].at[time_idx + 1].set(u_step)

        source_states = tuple(time_step_source(source_idx) for source_idx in range(len(sources)))

        return source_states

    def print_timestep(time_idx: int):
        if print_every:
            jax.lax.cond(
                time_idx % print_every == 0,
                lambda: jax.debug.print("Timestep {x}", x=time_idx),
                lambda: None,
            )

    def create_charges(time_idx: int, source_states: tuple[Array, ...]):
        def create_charge(source_idx: int, charge_idx: int):
            charge_0 = sources[source_idx].charges_0[charge_idx]
            position_0_fn = charge_0.position_fn
            source_state = source_states[source_idx]

            position_array = source_state[:, charge_idx, 0]
            velocity_array = source_state[:, charge_idx, 1]
            t_end = ts[time_idx]

            updated_position_fn = interpolate_position(
                ts, position_array, velocity_array, position_0_fn, t_end
            )
            charge = replace(charge_0, position_fn=updated_position_fn)

            return charge

        return tuple(
            tuple(create_charge(s_idx, c_idx) for c_idx, _ in enumerate(source.charges_0))
            for s_idx, source in enumerate(sources)
        )

    return simulate_fn


def rk4_step(term, t, u, dt, other_charges):
    """Perform a single RK4 (Runge-Kutta 4th order) time step.

    Parameters
    ----------
    term : Callable
        ODE function with signature ``(t, u, other_charges) -> du/dt``.
    t : float
        Current time.
    u : Array
        Current state array, shape ``(n_charges, 2, 3)`` for ``[position, velocity]``.
    dt : float
        Time step size.
    other_charges : list[Charge]
        List of Charge objects from other sources for field calculation.

    Returns
    -------
    Array
        Updated state ``u + dt * (weighted average of k1, k2, k3, k4)``.
    """
    k1 = term(t, u, other_charges)
    k2 = term(t + dt / 2, u + dt / 2 * k1, other_charges)
    k3 = term(t + dt / 2, u + dt / 2 * k2, other_charges)
    k4 = term(t + dt, u + dt * k3, other_charges)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
