"""Physical source models for charged particle systems.

This module provides the Source dataclass and factory functions for creating
common charged particle configurations like dipole oscillators and free particles.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pycharge import potentials_and_fields
from pycharge.charge import Charge
from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class Source:
    """Container for a charged particle system and its equations of motion.

    A Source combines a collection of charges with an ODE function that describes their
    coupled dynamics. Used as input to the simulate function for time-evolution.

    Attributes
    ----------
    charges_0 : Sequence[Charge]
        Initial Charge objects at t=0. Sequence of one or more charges.
    ode_func : Callable
        Callable with signature ``(t, u, other_charges) -> du/dt`` where:

        - t : Current time (scalar)
        - u : State array shape ``(n_charges, 2, 3)`` containing ``[[r0, v0], [r1, v1], ...]``
        - other_charges : List of Charge objects from other sources

        Returns: Time derivative du/dt with same shape as u.

    See Also
    --------
    dipole_source : Factory for harmonic dipole oscillators
    free_particle_source : Factory for free charged particles
    """

    charges_0: Sequence[Charge]
    ode_func: Callable


def dipole_source(d_0: Vector3, q: float, omega_0: float, m: float, origin: Vector3) -> Source:
    r"""Create a harmonic dipole oscillator source with radiation damping.

    Constructs a Source containing two charges of opposite sign that oscillate as a
    damped harmonic dipole. The dipole experiences radiation reaction (Abraham-Lorentz
    damping) and can interact with external fields from other charges.

    The equations of motion for the dipole moment :math:`\mathbf{d} = \mathbf{r}_+ - \mathbf{r}_-` are:

    .. math::

        \ddot{\mathbf{d}} + \gamma_0 \dot{\mathbf{d}} + \omega_0^2 \mathbf{d} = \frac{2q}{m} \mathbf{E}_{\text{ext}}

    where :math:`\gamma_0` is the radiation damping coefficient:

    .. math::

        \gamma_0 = \frac{2q^2\omega_0^2}{3\pi\epsilon_0 m c^3}

    Parameters
    ----------
    d_0 : Vector3
        Initial dipole separation vector :math:`\mathbf{d}_0 = [d_x, d_y, d_z]` in meters.
        Defines both magnitude and orientation of the dipole.
    q : float
        Magnitude of each charge in Coulombs. Charges are :math:`\pm q`.
    omega_0 : float
        Natural angular frequency of oscillation in rad/s.
    m : float
        Total mass of the dipole in kg. Each charge has mass m/2.
    origin : Vector3
        Center position :math:`\mathbf{r}_c = [x_c, y_c, z_c]` of the dipole in meters.

    Returns
    -------
    Source
        Source object with two charges and dipole ODE function.

    Notes
    -----
    - The dipole responds only to the field component along its polarization axis
    - Radiation damping causes energy loss over time
    - Initial conditions have charges at rest at :math:`\pm \mathbf{d}_0/2` from origin
    - Compatible with multi-source simulations for dipole-dipole interactions

    References
    ----------
    .. [1] Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.). Section 16.2-16.3.
    .. [2] Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics. Chapter 12.
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
    r"""Create a free charged particle source subject to the Lorentz force.

    Constructs a Source containing a single charge that moves according to the Lorentz
    force law from electromagnetic fields produced by other charges. The particle
    follows the equation of motion:

    .. math::

        m\frac{d\mathbf{v}}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})

    Parameters
    ----------
    position_0_fn : Callable[[Scalar], Vector3]
        Function specifying initial position :math:`\mathbf{r}_0(t)` for t <= t_start.
        Should have signature ``(t) -> [x, y, z]``.
    q : float
        Charge magnitude in Coulombs. Can be positive or negative.
    m : float
        Mass of the particle in kilograms.

    Returns
    -------
    Source
        Source object with one charge and Lorentz force ODE function.

    Notes
    -----
    - Particle responds to both E and B fields from other sources
    - If other_charges is empty, particle moves with constant velocity (free drift)
    - Initial velocity must be set in the state array passed to simulation
    - Compatible with arbitrary external field configurations

    References
    ----------
    .. [1] Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.). Section 12.1.
    .. [2] Griffiths, D. J. (2017). Introduction to Electrodynamics (4th ed.). Section 10.1.
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
