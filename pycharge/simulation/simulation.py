import jax.numpy as jnp

from pycharge import Charge


def simulation(sources, t):
    charge_0_list = [charge_0 for source in sources for charge_0 in source.charges_0]

    charge_position_map = {charge_0: jnp.zeros([len(t), 3]) for charge_0 in charge_0_list}
    for charge_0 in charge_0_list:  # Initialize charge positions
        charge_position_map[charge_0] = charge_position_map[charge_0].at[0].set(charge_0.position(0))

    charge_map = {
        charge_0: Charge(
            lambda time: charge_0.position(time)
            if time < t[0]
            else jnp.interp(time, t, charge_position_map[charge_0]),  # Problem... won't work with 3D
            charge_0.q,
        )
        for charge_0 in charge_0_list
    }

    for time_step in range(len(t) - 1):
        dt = t[time_step + 1] - t[time_step]

        for source in sources:
            source_charges = [charge_map[charge_0] for charge_0 in source.charges_0]
            other_charges = [
                charge_map[charge_0] for charge_0 in charge_0_list if charge_0 not in source.charges_0
            ]

            (updated_charge_positions, vel) = rk4_step(
                lambda time: source.func_ode(source_charges, other_charges, time),
                [updated_charge_positions, vel],  # Yeah this is bad...!
                t[time_step],
                dt,
            )

            for charge_0 in charge_0_list:  # Update charge positions
                charge_position_map[charge_0] = (
                    charge_position_map[charge_0].at[time_step].set(updated_charge_positions)
                )

    return charge_map.values()


def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    from scipy.constants import e

    from .source import Source, dipole_ode

    source = Source(
        charges_0=[Charge(lambda t: [-1e-9, 0, 0], e), Charge(lambda t: [1e-9, 0, 0], -e)],
        func_ode=dipole_ode(),
    )
