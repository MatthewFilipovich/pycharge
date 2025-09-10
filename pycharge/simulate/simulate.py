from functools import partial

import jax.numpy as jnp

from pycharge import Charge


def simulate(sources, t):
    t = jnp.asarray(t)
    state_map = {id(source): jnp.zeros([len(t), len(source.charges_0), 2, 3]) for source in sources}

    def get_other_charges(source):
        other_charges = []
        other_sources = [s for s in sources if s is not source]

        for other_source in other_sources:
            for charge_idx, charge_0 in enumerate(other_source.charges_0):
                charge = Charge(
                    lambda time: charge_0.position(time)
                    if time < t[0]
                    else jnp.interp(
                        time, t, state_map[id(other_source)][:, charge_idx]
                    ),  # Problem... won't work with 3D
                    charge_0.q,
                )
                other_charges.append(charge)

        return other_charges

    def rk4_step(source, time_step):
        f = partial(source.func_ode, other_charges=get_other_charges(source))
        state = state_map[id(source)][time_step]

        time = t[time_step]
        dt = t[time_step + 1] - t[time_step]

        k1 = f(time, state)
        k2 = f(time + dt / 2, state + dt / 2 * k1)
        k3 = f(time + dt / 2, state + dt / 2 * k2)
        k4 = f(time + dt, state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for time_step in range(len(t) - 1):
        for source in sources:
            updated_state = rk4_step(source, time_step)
            state_map[id(source)] = state_map[id(source)].at[time_step + 1].set(updated_state)

    return state_map.values()


if __name__ == "__main__":
    from scipy.constants import e, m_e

    from pycharge.simulate.sources import dipole_source

    dipole0 = dipole_source(
        positions_0=[jnp.array([0.0, 0.0, 1e-9]), lambda t: jnp.array([0.0, 0.0, -1e-9])],
        q=e,
        omega_0=1e9,
        m=m_e,
    )

    dipole1 = dipole_source(
        positions_0=[jnp.array([0.0, 1e-8, 1e-9]), lambda t: jnp.array([0.0, 1e-8, -1e-9])],
        q=e,
        omega_0=1e9,
        m=m_e,
    )

    output = simulate([dipole0, dipole1], jnp.linspace(0, 1e-9, 100))
    print(output)
