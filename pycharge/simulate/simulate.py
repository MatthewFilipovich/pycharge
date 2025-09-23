import jax
import jax.numpy as jnp
from tqdm import tqdm

from pycharge import Charge

from ..utils import make_cubic_hermite_spline


def simulate(sources, t_start, t_end, t_num):  # TODO: change to ts and allow non-uniform spacing
    t_start = float(t_start)
    ts = jnp.linspace(t_start, t_end, t_num)
    dt = ts[1] - ts[0]

    state_list = [jnp.zeros([t_num, len(source.charges_0), 2, 3]) for source in sources]
    for source_idx, source in enumerate(sources):
        for charge_idx, charge in enumerate(source.charges_0):
            initial_state = jnp.array([charge.position(t_start), jax.jacobian(charge.position)(t_start)])
            state_list[source_idx] = state_list[source_idx].at[0, charge_idx].set(initial_state)

    def create_charge(source_idx, charge_0_idx):
        charge_0 = sources[source_idx].charges_0[charge_0_idx]

        def charge_position(t):
            t_index = ((t - t_start) / dt).astype(int)
            x0 = ts[t_index]
            y0 = state_list[source_idx][t_index, charge_0_idx, 0]
            m0 = state_list[source_idx][t_index, charge_0_idx, 1]
            x1 = ts[t_index + 1]
            y1 = state_list[source_idx][t_index + 1, charge_0_idx, 0]
            m1 = state_list[source_idx][t_index + 1, charge_0_idx, 1]

            return jnp.where(
                t < t_start, charge_0.position(t), make_cubic_hermite_spline(x0, y0, m0, x1, y1, m1)(t)
            )

        return Charge(charge_position, charge_0.q)

    charge_list = [
        [create_charge(source_idx, charge_0_idx) for charge_0_idx, _ in enumerate(source.charges_0)]
        for source_idx, source in enumerate(sources)
    ]
    charge_set = {c for sublist in charge_list for c in sublist}

    for time_step, time in enumerate(tqdm(ts[:-1])):
        for source_idx, source in enumerate(sources):
            other_charges = charge_set - {c for c in charge_list[source_idx]}
            current_state = state_list[source_idx][time_step]

            updated_state = rk4_step(source.func_ode, time, time + dt, current_state, dt, other_charges)
            state_list[source_idx] = state_list[source_idx].at[time_step + 1].set(updated_state)

    return state_list, charge_list


def rk4_step(term, t0, t1, y0, dt, args):
    k1 = term(t0, y0, args)
    k2 = term(t0 + dt / 2, y0 + dt / 2 * k1, args)
    k3 = term(t0 + dt / 2, y0 + dt / 2 * k2, args)
    k4 = term(t0 + dt, y0 + dt * k3, args)
    return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.constants import e, m_e

    from pycharge.simulate.sources import dipole_source

    # jax.config.update("jax_enable_x64", True)

    dipole0 = dipole_source(
        positions_0=[lambda t: jnp.array([0.0, 0.0, 0.5e-9]), lambda t: jnp.array([0.0, 0.0, -0.5e-9])],
        q=e,
        omega_0=100e12 * 2 * jnp.pi,
        m=m_e,
    )

    dipole1 = dipole_source(
        positions_0=[lambda t: jnp.array([0.0, 80e-9, 0.5e-9]), lambda t: jnp.array([0.0, 80e-9, -0.5e-9])],
        q=e,
        omega_0=100e12 * 2 * jnp.pi,
        m=m_e,
    )

    num_steps = 10000
    dt = 1e-18
    t_end = num_steps * dt

    state_list, charge_list = (simulate)([dipole0, dipole1], 0, t_end, num_steps)
    plt.plot(state_list[0][:, 0, 0, 2])
    plt.plot(state_list[0][:, 1, 0, 2])
    plt.show()
