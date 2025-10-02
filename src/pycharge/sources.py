"""This module defines the Source class."""

from typing import Callable, NamedTuple, Sequence

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge.charge import Charge
from pycharge.electromagnetic_quantities import electric_field


class Source(NamedTuple):
    charges_0: Sequence[Charge]
    ode_func: Callable


def dipole_source(d0, q, omega_0, m, origin=(0, 0, 0), polarized=True):
    if isinstance(m, (int, float)):
        m = (m, m)
    m_eff = m[0] * m[1] / (m[0] + m[1])
    gamma_0 = 1 / (4 * jnp.pi * epsilon_0) * 2 * q**2 * omega_0**2 / (3 * m_eff * c**3)
    origin = jnp.array(origin)
    d0 = jnp.array(d0)

    polarization_direction = abs(d0 / jnp.linalg.norm(d0))

    positions_0 = [lambda t: origin + d0 / 2, lambda t: origin - d0 / 2]

    def dipole_ode_fn(time, state, other_charges, config):
        r0, v0 = state[0]
        r1, v1 = state[1]

        x, y, z = (r0 + r1) / 2
        E = electric_field(other_charges, config)(x, y, z, time) if other_charges else 0

        if polarized:
            E = E * polarization_direction

        dipole_r = r0 - r1
        dipole_v = v0 - v1
        dipole_a = q / m_eff * E - gamma_0 * dipole_v - omega_0**2 * dipole_r

        dr0_dt = v0
        dr1_dt = v1
        dv0_dt = dipole_a / 2
        dv1_dt = -dipole_a / 2

        out = jnp.asarray([[dr0_dt, dv0_dt], [dr1_dt, dv1_dt]])

        return out

    return Source(charges_0=(Charge(positions_0[0], q), Charge(positions_0[1], -q)), ode_func=dipole_ode_fn)
