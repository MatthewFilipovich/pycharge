import jax
import jax.numpy as jnp

from pycharge import Charge

from ..utils import make_cubic_hermite_spline


def create_charges(sources, state_list, ts):
    t_start = ts[0]
    t_end = ts[-1]

    def create_charge(source_idx, charge_0_idx):
        charge_0 = sources[source_idx].charges_0[charge_0_idx]

        def charge_position(t):
            def before_start():
                return charge_0.position(t)

            def interpolate():
                t_idx = ((t - t_start) / dt).astype(int)

                # Define the "if True" branch as a function
                def handle_nan():
                    return state_list[source_idx][-1, charge_0_idx, 0]

                # Define the "else" branch as a function
                def do_interpolation():
                    x0 = ts[t_idx]
                    y0 = state_list[source_idx][t_idx, charge_0_idx, 0]
                    m0 = state_list[source_idx][t_idx, charge_0_idx, 1]
                    x1 = ts[t_idx + 1]
                    y1 = state_list[source_idx][t_idx + 1, charge_0_idx, 0]
                    m1 = state_list[source_idx][t_idx + 1, charge_0_idx, 1]

                    spline_fn = make_cubic_hermite_spline(x0, y0, m0, x1, y1, m1)
                    return spline_fn(t)

                # The condition to check
                is_nan = jnp.any(jnp.isnan(state_list[source_idx][t_idx + 1, charge_0_idx]))

                # Use jax.lax.cond to select the correct branch
                return jax.lax.cond(is_nan, handle_nan, do_interpolation)

            def after_finish():
                return state_list[source_idx][-1, charge_0_idx, 0]

            return jax.lax.cond(
                t < t_start, before_start, lambda: jax.lax.cond(t < t_end, interpolate, after_finish)
            )

        return Charge(charge_position, charge_0.q)

    return [
        [create_charge(source_idx, charge_0_idx) for charge_0_idx, _ in enumerate(source.charges_0)]
        for source_idx, source in enumerate(sources)
    ]


def simulate(sources):
    def simulate_fn(ts):  # TODO: change to ts and allow non-uniform spacing
        t_num = len(ts)
        t_start = ts[0]
        dt = ts[1] - ts[0]

        # NOTE: THE + 1 is to save the very last value at each time idx... let's see if this works
        state_list = [jnp.full([t_num + 1, len(source.charges_0), 2, 3], jnp.nan) for source in sources]

        # Initial conditions
        for source_idx, source in enumerate(sources):
            for charge_idx, charge in enumerate(source.charges_0):
                initial_state = jnp.array([charge.position(t_start), jax.jacobian(charge.position)(t_start)])
                state_list[source_idx] = state_list[source_idx].at[0, charge_idx].set(initial_state)
                state_list[source_idx] = state_list[source_idx].at[-1, charge_idx].set(initial_state)

        # Charges
        charge_list = create_charges(sources, state_list, ts)
        charge_set = {c for sublist in charge_list for c in sublist}

        # ODEs
        func_ode_list = []
        for source_idx, source in enumerate(sources):
            other_charges = charge_set - {c for c in charge_list[source_idx]}
            func_ode_list.append(source.func_ode(other_charges))

        def time_step_body(time_idx, s):
            print(time_idx)
            jax.debug.print("{x}", x=time_idx)
            time = ts[time_idx]

            for source_idx, _ in enumerate(sources):
                current_s = s[source_idx][time_idx]
                updated_s = rk4_step(func_ode_list[source_idx], time, time + dt, current_s, dt)

                s[source_idx] = s[source_idx].at[time_idx + 1].set(updated_s)
                s[source_idx] = s[source_idx].at[-1].set(updated_s)

            return s

        final_states_with_buffer = jax.lax.fori_loop(0, t_num - 1, time_step_body, state_list)
        return [s[:-1] for s in final_states_with_buffer]

        # TODO: Define one scan step jax.lax.fori_loop
        # for time_idx, time in enumerate((ts[:-1])):
        #     print(time_idx)
        #     for source_idx, _ in enumerate(sources):
        #         current_state = state_list[source_idx][time_idx]
        #         updated_state = rk4_step(func_ode_list[source_idx], time, time + dt, current_state, dt)

        #         state_list[source_idx] = state_list[source_idx].at[time_idx + 1].set(updated_state)
        #         state_list[source_idx] = state_list[source_idx].at[-1].set(updated_state)

        # return [s[:-1] for s in state_list]  # Pop last element

    return simulate_fn


def rk4_step(term, t0, t1, y0, dt):
    k1 = term(t0, y0)
    k2 = term(t0 + dt / 2, y0 + dt / 2 * k1)
    k3 = term(t0 + dt / 2, y0 + dt / 2 * k2)
    k4 = term(t0 + dt, y0 + dt * k3)
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
    # sim_fn = jax.jit(sim_fn)
    # sim_fn = equinox.filter_jit(sim_fn)
    state_list = sim_fn(ts)
    plt.plot(state_list[0][:, 0, 0, 2])
    plt.plot(state_list[0][:, 1, 0, 2])
    plt.show()
    plt.plot(state_list[1][:, 0, 0, 2])
    plt.plot(state_list[1][:, 1, 0, 2])
    plt.show()
