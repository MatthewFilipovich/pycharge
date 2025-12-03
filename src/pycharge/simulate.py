from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from pycharge.functional import interpolate_position, position, velocity
from pycharge.sources import Source


def simulate(
    sources: Sequence[Source], ts: Array, print_every_n_timesteps: int = 100
) -> Callable[[], tuple[Array, ...]]:
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
    k1 = term(t, u, other_charges)
    k2 = term(t + dt / 2, u + dt / 2 * k1, other_charges)
    k3 = term(t + dt / 2, u + dt / 2 * k2, other_charges)
    k4 = term(t + dt, u + dt * k3, other_charges)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
