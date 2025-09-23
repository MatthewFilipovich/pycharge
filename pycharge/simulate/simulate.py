import equinox
import jax
import jax.numpy as jnp

from pycharge import Charge

from ..utils import make_cubic_hermite_spline


def simulate(sources):
    def simulate_fn(ts):  # TODO: change to ts and allow non-uniform spacing
        t_num = len(ts)
        t_start = ts[0]
        t_finish = ts[-1]
        dt = ts[1] - ts[0]

        # Preallocate state_list (we’ll treat it as a pytree of arrays)
        state_list = [jnp.zeros([t_num, len(source.charges_0), 2, 3]) for source in sources]

        # Initial conditions
        for source_idx, source in enumerate(sources):
            for charge_idx, charge in enumerate(source.charges_0):
                initial_state = jnp.array([charge.position(t_start), jax.jacobian(charge.position)(t_start)])
                state_list[source_idx] = state_list[source_idx].at[0, charge_idx].set(initial_state)

        # Charges
        def create_charge(source_idx, charge_0_idx):
            charge_0 = sources[source_idx].charges_0[charge_0_idx]

            def charge_position(t):
                def before_start():
                    return charge_0.position(t)

                def interpolate():
                    idx = ((t - t_start) / dt).astype(int)
                    x0 = ts[idx]
                    y0 = state_list[source_idx][idx, charge_0_idx, 0]
                    m0 = state_list[source_idx][idx, charge_0_idx, 1]
                    x1 = ts[idx + 1]
                    y1 = state_list[source_idx][idx + 1, charge_0_idx, 0]
                    m1 = state_list[source_idx][idx + 1, charge_0_idx, 1]

                    spline_fn = make_cubic_hermite_spline(x0, y0, m0, x1, y1, m1)
                    return spline_fn(t)

                def after_finish():
                    return state_list[source_idx][-1, charge_0_idx, 0]

                return jax.lax.cond(
                    t < t_start, before_start, lambda: jax.lax.cond(t < t_finish, interpolate, after_finish)
                )

            return Charge(charge_position, charge_0.q)

        charge_list = [
            [create_charge(source_idx, charge_0_idx) for charge_0_idx, _ in enumerate(source.charges_0)]
            for source_idx, source in enumerate(sources)
        ]
        charge_set = {c for sublist in charge_list for c in sublist}

        # ODEs
        func_ode_list = []
        for source_idx, source in enumerate(sources):
            other_charges = charge_set - {c for c in charge_list[source_idx]}
            func_ode_list.append(source.func_ode(other_charges))

        # Define one scan step
        def step_fn(carry, time_idx):
            jax.debug.print("{x}", x=time_idx)
            state_list = carry
            time = ts[time_idx]

            new_state_list = []
            for source_idx, source in enumerate(sources):
                current_state = state_list[source_idx][time_idx]
                updated_state = rk4_step(func_ode_list[source_idx], time, time + dt, current_state, dt)

                updated_src_state = state_list[source_idx].at[time_idx + 1].set(updated_state)
                new_state_list.append(updated_src_state)
                # jax.debug.print("new_state_list={x}", x=new_state_list)

            return new_state_list, None

        # Run scan over time steps
        state_list, _ = jax.lax.scan(step_fn, state_list, jnp.arange(t_num - 1))  # TODO: Should this be -2?

        # carry = state_list
        # for time_idx in tqdm(range(t_num - 1)):  # using Python int range for clarity
        #     carry, _ = step_fn(carry, time_idx)
        # state_list = carry

        return state_list

    return simulate_fn


def rk4_step(term, t0, t1, y0, dt):
    # jax.debug.print("y0={k1}", k1=y0)
    k1 = term(t0, y0)
    # jax.debug.print("{k1}", k1=k1)
    k2 = term(t0 + dt / 2, y0 + dt / 2 * k1)
    k3 = term(t0 + dt / 2, y0 + dt / 2 * k2)
    k4 = term(t0 + dt, y0 + dt * k3)
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

    num_steps = 40_000
    dt = 1e-18
    t_end = num_steps * dt
    ts = jnp.linspace(0, t_end, num_steps)

    sim_fn = simulate([dipole0, dipole1])
    # sim_fn = jax.jit(sim_fn)
    # sim_fn = equinox.filter_jit(sim_fn)
    state_list = sim_fn(ts)
    plt.plot(state_list[0][:, 0, 0, 2])
    plt.plot(state_list[0][:, 1, 0, 2])
    plt.show()
