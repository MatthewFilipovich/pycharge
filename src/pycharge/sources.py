from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge import potentials_and_fields
from pycharge.charge import Charge
from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class Source:
    charges_0: Sequence[Charge]
    ode_func: Callable


def dipole_source(d_0: Vector3, q: float, omega_0: float, m: float, origin: Vector3) -> Source:
    d_0 = jnp.asarray(d_0)
    origin = jnp.asarray(origin)

    m_eff = m / 2
    gamma_0 = 1 / (4 * jnp.pi * epsilon_0) * 2 * q**2 * omega_0**2 / (3 * m_eff * c**3)
    polarization_direction = jnp.abs(d_0 / jnp.linalg.norm(d_0))

    positions_0 = [lambda t: origin - d_0 / 2, lambda t: origin + d_0 / 2]

    def dipole_ode_fn(t, u, other_charges):
        r0, v0 = u[0]
        r1, v1 = u[1]

        x, y, z = (r0 + r1) / 2
        E = potentials_and_fields(other_charges)(x, y, z, t).electric if other_charges else 0
        E = E * polarization_direction

        dipole_r = r1 - r0
        dipole_v = v1 - v0
        dipole_a = q / m_eff * E - gamma_0 * dipole_v - omega_0**2 * dipole_r

        dr0_dt, dv0_dt = v0, -dipole_a / 2
        dr1_dt, dv1_dt = v1, dipole_a / 2

        return jnp.array([[dr0_dt, dv0_dt], [dr1_dt, dv1_dt]])

    return Source(charges_0=(Charge(positions_0[0], -q), Charge(positions_0[1], q)), ode_func=dipole_ode_fn)


def free_particle_source(position_0_fn: Callable[[Scalar], Vector3], q: float, m: float) -> Source:
    def free_particle_ode_fn(t, u, other_charges):
        r, v = u[0]
        x, y, z = r

        quantities = potentials_and_fields(other_charges)(x, y, z, t) if other_charges else None
        E = quantities.electric if quantities else jnp.zeros(3)
        B = quantities.magnetic if quantities else jnp.zeros(3)

        dr_dt = v
        dv_dt = q / m * (E + jnp.cross(v, B))

        return jnp.array([[dr_dt, dv_dt]])

    return Source(charges_0=(Charge(position_0_fn, q),), ode_func=free_particle_ode_fn)
