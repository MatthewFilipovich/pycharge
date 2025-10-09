from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from pycharge.charge import Charge
from pycharge.sources import Source
from pycharge.utils import interpolate_position


def simulate(sources: Sequence[Source], print_every_n_timesteps: int = 100):
    def simulate_fn(ts: Array):
        initial_state = tuple(create_initial_state(source, ts) for source in sources)
        final_state = jax.lax.fori_loop(0, len(ts) - 1, time_step_body, initial_state)

        return final_state

    def create_initial_state(source: Source, ts: Array) -> Array:
        state_array = jnp.full([len(ts), len(source.charges_0), 2, 3], jnp.nan)

        for charge_idx, charge in enumerate(source.charges_0):
            pos0, vel0 = charge.position(ts[0]), jax.jacobian(charge.position)(ts[0])
            state_array = state_array.at[0, charge_idx].set([pos0, vel0])
        return state_array

    def time_step_body(time_idx: int, state: tuple[Array, ...]) -> tuple[Array, ...]:
        if print_every_n_timesteps:
            jax.lax.cond(
                time_idx % print_every_n_timesteps == 0,
                lambda: jax.debug.print("Timestep {x}", x=time_idx),
                lambda: None,
            )

        def create_charge(source_idx: int, charge_idx: int) -> Charge:
            charge0 = sources[source_idx].charges_0[charge_idx]
            position_0 = charge0.position
            position_array = state[source_idx][:, charge_idx, 0]
            velocity_array = state[source_idx][:, charge_idx, 1]
            nan_position_fill_value = state[source_idx][time_idx, charge_idx, 0]

            position = interpolate_position(
                ts, position_array, velocity_array, position_0, nan_position_fill_value
            )
            q = charge0.q

            return Charge(position, q)

        charges = tuple(
            tuple(create_charge(s_idx, c_idx) for c_idx, _ in enumerate(source.charges_0))
            for s_idx, source in enumerate(sources)
        )

        t0 = ts[time_idx]
        t1 = ts[time_idx + 1]
        dt = t1 - t0

        for source_idx, source in enumerate(sources):
            other_charges = tuple(ch for i, src in enumerate(charges) if i != source_idx for ch in src)
            current_s = state[source_idx][time_idx]
            updated_s = rk4_step(source.ode_func, t0, t1, current_s, dt, other_charges)

            state = (
                state[:source_idx]
                + (state[source_idx].at[time_idx + 1].set(updated_s),)
                + state[source_idx + 1 :]
            )

        return state

    return simulate_fn


def rk4_step(term, t0, t1, y0, dt, other_charges):
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
