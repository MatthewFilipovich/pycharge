from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from pycharge.sources import Source
from pycharge.utils import interpolate_position


def simulate(
    sources: Sequence[Source], print_every_n_timesteps: int = 100
) -> Callable[[Array], tuple[Array, ...]]:
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
        print_every_n_timesteps: If positive, prints the current timestep every
            `n` steps during the simulation. Useful for debugging long runs.
            Defaults to 100.

    Returns:
        A function that executes the simulation. This function takes a single
        argument ``ts`` (a JAX array of time steps) and returns a tuple of
        state arrays, one for each source. Each state array has the shape
        ``(num_timesteps, num_charges_in_source, 2, 3)``, where the last two
        dimensions correspond to the position and velocity vectors of each charge.
    """

    def simulate_fn(ts: Array):
        source_states = tuple(create_initial_state(ts, source) for source in sources)
        source_states = jax.lax.fori_loop(0, len(ts) - 1, time_step_body, source_states)

        return source_states

    def create_initial_state(ts: Array, source: Source) -> Array:
        source_state = jnp.full([len(ts), len(source.charges_0), 2, 3], jnp.nan)
        for charge_idx, charge in enumerate(source.charges_0):
            pos0, vel0 = charge.position(ts[0]), jax.jacobian(charge.position)(ts[0])
            source_state = source_state.at[0, charge_idx, :, :].set([pos0, vel0])

        return source_state

    def time_step_body(time_idx: int, source_states: tuple[Array, ...]) -> tuple[Array, ...]:
        print_timestep(time_idx)

        charges = create_charges(time_idx, source_states)
        t = ts[time_idx]
        dt = ts[time_idx + 1] - ts[time_idx]

        def time_step_source(source_idx):
            u = source_states[source_idx][time_idx]
            other_charges_flat = [c for i, c_tuple in enumerate(charges) if i != source_idx for c in c_tuple]
            ode_func = sources[source_idx].ode_func
            y_step = rk4_step(ode_func, t, u, dt, other_charges_flat)

            return source_states[source_idx].at[time_idx + 1].set(y_step)

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
            position_0 = charge_0.position
            source_state = source_states[source_idx]

            position_array = source_state[:, charge_idx, 0]
            velocity_array = source_state[:, charge_idx, 1]
            t_end = ts[time_idx]

            updated_position = interpolate_position(ts, position_array, velocity_array, position_0, t_end)
            charge = replace(charge_0, position=updated_position)

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


if __name__ == "__main__":
    from time import time

    import matplotlib.pyplot as plt
    from scipy.constants import e, m_e

    from pycharge.sources import dipole_source

    # jax.config.update("jax_enable_x64", True)

    dipole0 = dipole_source(d_0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e)
    dipole1 = dipole_source(
        d_0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 80e-9, 0.0]
    )

    t_start = 0.0
    t_num = 40_000
    dt = 1e-18
    t_stop = (t_num - 1) * dt
    ts = jnp.linspace(t_start, t_stop, t_num)

    sim_fn = simulate([dipole0, dipole1])
    sim_fn = jax.jit(sim_fn)
    # sim_fn = equinox.filter_jit(sim_fn)

    start_time = time()
    state_list = sim_fn(ts)
    print("Time:", time() - start_time)
    start_time = time()
    state_list = sim_fn(ts)
    print("Time:", time() - start_time)

    plt.plot(state_list[0][:, 0, 0, 2])
    plt.plot(state_list[0][:, 1, 0, 2])
    plt.show()
    plt.plot(state_list[1][:, 0, 0, 2])
    plt.plot(state_list[1][:, 1, 0, 2])
    plt.show()
