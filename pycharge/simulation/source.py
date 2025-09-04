"""This module defines the Source class."""

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeAlias

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.constants import c, epsilon_0, m_e

from pycharge import Charge, electric_field

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]


@dataclass(frozen=True)
class Source:
    charges_0: Iterable[Charge]
    func_ode: Callable  # For each charge!


def dipole_ode(omega_0, m=m_e):
    m_eff = m / 2

    def dipole_ode_fn(source_charges, other_charges, time):
        # Calculate dipole position and velocity
        r = jnp.mean(source_charges[0].position(time), source_charges[1].position(time))
        v = jnp.mean(
            jax.jacobian(source_charges[0].position)(time), jax.jacobian(source_charges[1].position)(time)
        )
        E = electric_field(other_charges)(r[0], r[1], r[2], time)
        q = source_charges[0].q  # If they are different??
        gamma_0 = 1 / (4 * jnp.pi * epsilon_0) * 2 * q**2 * omega_0**2 / (3 * m_eff * c**3)
        dx_dt = v
        dv_dt = q / m_eff * E - gamma_0 * v - omega_0 * r

        return dx_dt, dv_dt

    return dipole_ode_fn
