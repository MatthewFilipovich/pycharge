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
        t0, t1 = ts[time_idx], ts[time_idx + 1]
        dt = t1 - t0

        def time_step_source(source_idx):
            y0 = source_states[source_idx][time_idx]
            other_charges_flat = [c for i, c_tuple in enumerate(charges) if i != source_idx for c in c_tuple]
            ode_func = sources[source_idx].ode_func
            y_step = rk4_step(ode_func, t0, y0, dt, other_charges_flat)

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


def rk4_step(term, t0, y0, dt, other_charges):
    k1 = term(t0, y0, other_charges)
    k2 = term(t0 + dt / 2, y0 + dt / 2 * k1, other_charges)
    k3 = term(t0 + dt / 2, y0 + dt / 2 * k2, other_charges)
    k4 = term(t0 + dt, y0 + dt * k3, other_charges)
    return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    from time import time

    import matplotlib.pyplot as plt
    from scipy.constants import e, m_e

    from pycharge.sources import dipole_source

    # jax.config.update("jax_enable_x64", True)

    dipole0 = dipole_source(d0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e)
    dipole1 = dipole_source(
        d0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 80e-9, 0.0]
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
