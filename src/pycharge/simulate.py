from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from pycharge.sources import Source
from pycharge.functional import interpolate_position, position, velocity


def simulate(
    sources: Sequence[Source], ts: Array, print_every_n_timesteps: int = 100
) -> Callable[[], tuple[Array, ...]]:
    """Factory function that creates a JAX-jittable simulation function.

    This function sets up a simulation for a system of sources (e.g., dipoles,
    free particles) whose dynamics are governed by an ordinary differential
    equation (ODE). It returns a new function that, when called with a time array,
    runs the simulation.

    The simulation proceeds by time-stepping using the 4th-order Runge-Kutta (RK4)
    method. At each step, the driving fields for each source are calculated from
    all other sources in the system.

    Args:
        sources: A sequence of ``Source`` objects to be simulated.
        ts: A JAX array of time steps at which to evaluate the simulation.
        print_every_n_timesteps: If positive, prints the current timestep every
            `n` steps during the simulation. Useful for debugging long runs.
            Defaults to 100.

    Returns:
        A function that executes the simulation. This function takes no arguments and returns a tuple of
        state arrays, one for each source. Each state array has the shape
        ``(num_timesteps, num_charges_in_source, 2, 3)``, where the last two
        dimensions correspond to the position and velocity vectors of each charge.
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
        if print_every_n_timesteps:
            jax.lax.cond(
                time_idx % print_every_n_timesteps == 0,
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
    """Performs a single 4th-order Runge-Kutta (RK4) step.

    This function advances the solution of an ordinary differential equation
    (ODE) from time `t` to `t + dt`.

    Args:
        term: The function that defines the ODE, i.e., dy/dt = term(t, y, ...).
        t: The current time.
        u: The current state of the system.
        dt: The time step size.
        other_charges: A list of other charges in the system, passed to the
            `term` function to calculate driving fields.

    Returns:
        The new state of the system at time `t + dt`.
    """
    k1 = term(t, u, other_charges)
    k2 = term(t + dt / 2, u + dt / 2 * k1, other_charges)
    k3 = term(t + dt / 2, u + dt / 2 * k2, other_charges)
    k4 = term(t + dt, u + dt * k3, other_charges)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
