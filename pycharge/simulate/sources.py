"""This module defines the Source class."""

from typing import Callable, Iterable, NamedTuple

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge import Charge, electric_field


class Source(NamedTuple):
    charges_0: Iterable[Charge]
    func_ode: Callable  # For each charge!


def dipole_source(positions_0, q, omega_0, m):
    if isinstance(m, (int, float)):
        m = (m, m)
    m_eff = m[0] * m[1] / (m[0] + m[1])
    gamma_0 = 1 / (4 * jnp.pi * epsilon_0) * 2 * q**2 * omega_0**2 / (3 * m_eff * c**3)

    def dipole_ode_fn(other_charges):
        electric_field_fn = (electric_field(other_charges)) if other_charges else 0

        def fn(time, state):
            r0, v0 = state[0]
            r1, v1 = state[1]

            dipole_origin = (r0 + r1) / 2
            E = electric_field_fn(*dipole_origin, time) if other_charges else 0

            dipole_r = r0 - r1
            dipole_v = v0 - v1
            dipole_a = q / m_eff * E - gamma_0 * dipole_v - omega_0**2 * dipole_r

            dr0_dt = v0
            dr1_dt = v1
            dv0_dt = dipole_a / 2
            dv1_dt = -dipole_a / 2

            out = jnp.asarray([[dr0_dt, dv0_dt], [dr1_dt, dv1_dt]])

            return out

        return fn

    return Source(charges_0=(Charge(positions_0[0], q), Charge(positions_0[1], -q)), func_ode=dipole_ode_fn)
