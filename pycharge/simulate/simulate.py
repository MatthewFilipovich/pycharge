import jax
import jax.numpy as jnp

from pycharge import Charge

from .interpolate_position import interpolate_position


def simulate(sources):
    def simulate_fn(ts):  # TODO: change to ts and allow non-uniform spacing
        t_num = len(ts)
        t_start = ts[0]
        dt = ts[1] - ts[0]

        def create_initial_state(source):
            state_array = jnp.full([t_num, len(source.charges_0), 2, 3], jnp.nan)

            for charge_idx, charge in enumerate(source.charges_0):
                pos0, vel0 = charge.position(t_start), jax.jacobian(charge.position)(t_start)
                state_array = state_array.at[0, charge_idx].set([pos0, vel0])
            return state_array

        def time_step_body(time_idx, state):
            def create_charge(source_idx, charge_idx):
                charge0 = sources[source_idx].charges_0[charge_idx]
                position_0 = charge0.position
                position_array = state[source_idx][:, charge_idx, 0]
                velocity_array = state[source_idx][:, charge_idx, 1]
                default_nan_position = state[source_idx][time_idx, charge_idx, 0]

                position = interpolate_position(
                    ts, position_0, position_array, velocity_array, default_nan_position
                )
                q = charge0.q

                return Charge(position, q)

            jax.debug.print("{x}", x=time_idx)

            charges = tuple(
                tuple(create_charge(s_idx, c_idx) for c_idx, _ in enumerate(source.charges_0))
                for s_idx, source in enumerate(sources)
            )

            time = ts[time_idx]
            for source_idx, source in enumerate(sources):
                other_charges = tuple(ch for i, src in enumerate(charges) if i != source_idx for ch in src)
                current_s = state[source_idx][time_idx]
                updated_s = rk4_step(source.ode_func, time, time + dt, current_s, dt, other_charges)

                state = (
                    state[:source_idx]
                    + (state[source_idx].at[time_idx + 1].set(updated_s),)
                    + state[source_idx + 1 :]
                )

            return state

        initial_state = tuple(create_initial_state(source) for source in sources)
        final_state = jax.lax.fori_loop(0, t_num - 1, time_step_body, initial_state)

        return final_state

    return simulate_fn


def rk4_step(term, t0, t1, y0, dt, other_charges):
    k1 = term(t0, y0, other_charges)
    k2 = term(t0 + dt / 2, y0 + dt / 2 * k1, other_charges)
    k3 = term(t0 + dt / 2, y0 + dt / 2 * k2, other_charges)
    k4 = term(t0 + dt, y0 + dt * k3, other_charges)
    return y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.constants import e, m_e

    from pycharge.simulate.sources import dipole_source

    # jax.config.update("jax_enable_x64", True)

    dipole0 = dipole_source(d0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e)
    dipole1 = dipole_source(
        d0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 80e-9, 0.0]
    )

    num_steps = 40_000
    dt = 1e-18
    t_end = (num_steps - 1) * dt
    ts = jnp.linspace(0, t_end, num_steps)

    sim_fn = simulate([dipole0, dipole1])
    sim_fn = jax.jit(sim_fn)
    # sim_fn = equinox.filter_jit(sim_fn)
    state_list = sim_fn(ts)
    plt.plot(state_list[0][:, 0, 0, 2])
    plt.plot(state_list[0][:, 1, 0, 2])
    plt.show()
    plt.plot(state_list[1][:, 0, 0, 2])
    plt.plot(state_list[1][:, 1, 0, 2])
    plt.show()
