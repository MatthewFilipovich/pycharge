"""Physical source models for charged particle systems."""

from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge import potentials_and_fields
from pycharge.charge import Charge
from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class Source:
    """Electromagnetic source used in electrodynamics simulations.

    Encapsulates charges and equations of motion for use with :func:`~pycharge.simulate`.

    Attributes:
        charges_0 (Sequence[Charge]): Initial charges at t=0.
        ode_func (Callable): ODE function ``(t, u, other_charges) -> du/dt`` where u has shape
            ``(n_charges, 2, 3)`` for ``[[r0, v0], [r1, v1], ...]``.
    """

    charges_0: Sequence[Charge]
    ode_func: Callable


def dipole_source(d_0: Vector3, q: float, omega_0: float, m: float, origin: Vector3) -> Source:
    r"""Create harmonic dipole oscillator with radiation damping.

    Two charges :math:`\pm q` oscillate as damped harmonic dipole with radiation reaction.
    Equation of motion for dipole moment :math:`\mathbf{d} = \mathbf{r}_+ - \mathbf{r}_-`:

    .. math::

        \ddot{\mathbf{d}} + \gamma_0 \dot{\mathbf{d}} + \omega_0^2 \mathbf{d} = \frac{2q}{m} \mathbf{E}_{\text{ext}}

    where :math:`\gamma_0 = \frac{2q^2\omega_0^2}{3\pi\epsilon_0 m c^3}` is the damping coefficient.

    Args:
        d_0 (Vector3): Initial dipole separation :math:`\mathbf{d}_0` (m).
        q (float): Charge magnitude (C) of each charge. Charges are :math:`+q` and :math:`-q`.
        omega_0 (float): Natural frequency (rad/s).
        m (float): Mass (kg) of each charge. Effective (reduced) mass is :math:`m/2`.
        origin (Vector3): Center position (m).

    Returns:
        Source: Source with two charges and dipole ODE.

    Note:
        Dipole responds only to field along polarization axis.
    """
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
    r"""Create free charged particle subject to Lorentz force.

    Particle evolves according to :math:`m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})`.

    Args:
        position_0_fn (Callable[[Scalar], Vector3]): Initial position function :math:`\mathbf{r}_0(t)`.
        q (float): Charge (C).
        m (float): Mass (kg).

    Returns:
        Source: Source with one charge and Lorentz force ODE.

    Note:
        If no other charges present, particle moves with constant velocity.
    """

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
