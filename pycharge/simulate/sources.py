"""This module defines the Source class."""

from dataclasses import dataclass
from typing import Callable, Iterable

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge import Charge, electric_field


@dataclass(frozen=True)
class Source:
    charges_0: Iterable[Charge]
    func_ode: Callable  # For each charge!


def dipole_source(positions_0, q, omega_0, m):
    m_eff = m[0] * m[1] / (m[0] + m[1])
    gamma_0 = 1 / (4 * jnp.pi * epsilon_0) * 2 * q**2 * omega_0**2 / (3 * m_eff * c**3)

    def dipole_ode_fn(time, state, other_charges):
        r, v = state[0]
        E = electric_field(other_charges)(r[0], r[1], r[2], time) if other_charges else 0

        dx_dt = v
        dv_dt = q / m_eff * E - gamma_0 * v - omega_0 * r

        out = jnp.asarray([[dx_dt, dv_dt], [-dx_dt, -dv_dt]])

        return out

    return Source(charges_0=(Charge(positions_0[0], q), Charge(positions_0[1], -q)), func_ode=dipole_ode_fn)
