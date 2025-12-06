"""Electromagnetic potentials and fields from moving point charges.

This module implements the Liénard-Wiechert potentials and their derivatives to compute
electromagnetic fields (E and B) from arbitrarily moving point charges. The calculations
account for retardation effects and relativistic corrections.
"""

from typing import Callable, Iterable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy.constants import c, epsilon_0, pi

from pycharge.charge import Charge
from pycharge.functional import acceleration, position, source_time, velocity


class Quantities(NamedTuple):
    r"""Container for electromagnetic quantities computed from moving charges.

    This named tuple holds the scalar potential, vector potential, electric and magnetic
    fields, along with decomposed field components that separate velocity-dependent and
    acceleration-dependent contributions.

    Attributes
    ----------
    scalar : Array
        Scalar potential :math:`\phi` (Liénard-Wiechert potential), shape ``(...)``.
    vector : Array
        Vector potential :math:`\mathbf{A}`, shape ``(..., 3)``.
    electric : Array
        Total electric field :math:`\mathbf{E} = \mathbf{E}_1 + \mathbf{E}_2`, shape ``(..., 3)``.
    magnetic : Array
        Total magnetic field :math:`\mathbf{B} = \mathbf{B}_1 + \mathbf{B}_2`, shape ``(..., 3)``.
    electric_term1 : Array
        Velocity-dependent (Coulomb-like) electric field component :math:`\mathbf{E}_1`, shape ``(..., 3)``.
    electric_term2 : Array
        Acceleration-dependent (radiation) electric field component :math:`\mathbf{E}_2`, shape ``(..., 3)``.
    magnetic_term1 : Array
        Velocity-dependent magnetic field component :math:`\mathbf{B}_1 = \mathbf{n} \times \mathbf{E}_1 / c`, shape ``(..., 3)``.
    magnetic_term2 : Array
        Acceleration-dependent (radiation) magnetic field component :math:`\mathbf{B}_2 = \mathbf{n} \times \mathbf{E}_2 / c`, shape ``(..., 3)``.

    Notes
    -----
    All fields are evaluated at the retarded time, accounting for the finite speed
    of light propagation from the source charge to the observation point.

    - :math:`\mathbf{E}_1` falls off as :math:`1/R^2` (near field)
    - :math:`\mathbf{E}_2` falls off as :math:`1/R` (radiation field)
    """

    scalar: Array
    vector: Array
    electric: Array
    magnetic: Array

    electric_term1: Array
    electric_term2: Array
    magnetic_term1: Array
    magnetic_term2: Array


def potentials_and_fields(
    charges: Iterable[Charge],
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Quantities]:
    r"""Create a function to compute electromagnetic potentials and fields from moving charges.

    Constructs a function that calculates the Liénard-Wiechert potentials and electromagnetic
    fields (E and B) at specified spacetime points due to one or more moving point charges.
    The implementation accounts for retardation effects and supports arbitrary charge motions.

    The electric and magnetic fields are computed using:

    .. math::

        \mathbf{E}(\mathbf{r}, t)
        = \frac{q}{4\pi\varepsilon_0} \left[
          \frac{\mathbf{n}_s - \boldsymbol{\beta}_s}{\gamma^{2}(1-\boldsymbol{\beta}_s \cdot \mathbf{n}_s)^{3}\,R^{2}} \;+\;
          \frac{\mathbf{n}_s \times \bigl\{(\mathbf{n}_s - \boldsymbol{\beta}_s) \times \dot{\boldsymbol{\beta}}_s\bigr\}}{c\,(1-\boldsymbol{\beta}_s \cdot \mathbf{n}_s)^{3}\,R}
        \right]_{t_r}

    .. math::

        \mathbf{B}(\mathbf{r}, t)
        = \frac{\mathbf{n}_s(t_r)}{c} \times \mathbf{E}(\mathbf{r}, t)

    where all source quantities are evaluated at the retarded time :math:`t_r`. The first term in
    the electric field expression is the velocity-dependent term (scaling as :math:`1/R^2`), and
    the second term is the acceleration-dependent radiation term (scaling as :math:`1/R`). These
    individual terms can be accessed via ``quantities.electric_term1`` and ``quantities.electric_term2``,
    respectively. Similarly, the magnetic field terms are accessible via ``quantities.magnetic_term1``
    and ``quantities.magnetic_term2``.

    Parameters
    ----------
    charges : Charge or Iterable[Charge]
        Single Charge object or iterable of Charge objects.

    Returns
    -------
    Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Quantities]
        A function ``quantities_fn(x, y, z, t)`` that computes electromagnetic quantities.
        The returned function takes:

        - x, y, z : Spatial coordinates (scalars or arrays of the same shape)
        - t : Time coordinate (scalar or array matching x, y, z shape)

        And returns a Quantities namedtuple containing all electromagnetic fields.

    Raises
    ------
    ValueError
        If charges is empty or if x, y, z, t don't have matching shapes.

    Notes
    -----
    - All calculations use the retarded time, accounting for light-speed propagation
    - Supports vectorized evaluation over arbitrary arrays of points
    - Uses JAX for automatic differentiation and JIT compilation
    - The function is compatible with JAX transformations (jit, vmap, grad)

    References
    ----------
    .. [1] Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.). Wiley.
    .. [2] Griffiths, D. J. (2017). Introduction to Electrodynamics (4th ed.). Cambridge.
    """
    charges = [charges] if isinstance(charges, Charge) else list(charges)

    if len(charges) == 0:
        raise ValueError("At least one charge must be provided.")

    qs = jnp.array([charge.q for charge in charges])

    def quantities_fn(x: ArrayLike, y: ArrayLike, z: ArrayLike, t: ArrayLike) -> Quantities:
        x, y, z, t = jnp.asarray(x), jnp.asarray(y), jnp.asarray(z), jnp.asarray(t)
        if not (x.shape == y.shape == z.shape == t.shape):
            raise ValueError("x, y, z, and t must have the same shape.")
        original_shape = x.shape

        r = jnp.stack([x, y, z], axis=-1)  # Stack into (..., 3)
        r_flat, t_flat = r.reshape(-1, 3), t.ravel()  # Flatten for vmap

        quantities_flat = jax.vmap(calculate_total_sources)(r_flat, t_flat)
        scalar_flat, other_quantities_flat = quantities_flat[0], quantities_flat[1:]

        return Quantities(
            scalar_flat.reshape(original_shape),  # 1D scalar quantity
            *(q.reshape(*original_shape, 3) for q in other_quantities_flat),  # 3D vectors quantities
        )

    def calculate_total_sources(r: Array, t: Array) -> Quantities:
        # Solve for retarded times for each charge
        t_srcs = jnp.stack([source_time(r, t, charge) for charge in charges])
        # Evaluate source properties at the retarded times
        r_srcs = jnp.stack([position(tr, charge) for tr, charge in zip(t_srcs, charges)])
        v_srcs = jnp.stack([velocity(tr, charge) for tr, charge in zip(t_srcs, charges)])
        a_srcs = jnp.stack([acceleration(tr, charge) for tr, charge in zip(t_srcs, charges)])

        # Compute individual contributions
        calculate_individual_source_vmap = jax.vmap(calculate_individual_source, in_axes=(0, 0, 0, 0, None))
        individual_quantities = calculate_individual_source_vmap(r_srcs, v_srcs, a_srcs, qs, r)
        # Sum contributions from all charges
        summed_quantities = Quantities(*(jnp.sum(value, axis=0) for value in individual_quantities))

        return summed_quantities

    def calculate_individual_source(
        r_src: Array, v_src: Array, a_src: Array, q: Array, r: Array
    ) -> Quantities:
        R = jnp.linalg.norm(r - r_src)  # Distance from source to observation point
        n = (r - r_src) / R  # Unit vector from source to observation
        β = v_src / c  # Velocity normalized to speed of light
        β̇ = a_src / c  # Acceleration normalized to speed of light
        one_minus_β_dot_β = 1 - jnp.dot(β, β)
        one_minus_n_dot_β = 1 - jnp.dot(n, β)
        one_minus_n_dot_β_cubed = one_minus_n_dot_β**3
        n_minus_β = n - β
        coeff = q / (4 * pi * epsilon_0)  # Common coefficient

        # Potentials
        φ = coeff / (one_minus_n_dot_β * R)
        A = β * φ / c

        # Fields
        E1 = coeff * n_minus_β * one_minus_β_dot_β / (one_minus_n_dot_β_cubed * R**2)
        E2 = coeff * jnp.cross(n, jnp.cross(n_minus_β, β̇)) / (c * one_minus_n_dot_β_cubed * R)
        E = E1 + E2

        B1 = jnp.cross(n, E1) / c
        B2 = jnp.cross(n, E2) / c
        B = B1 + B2

        return Quantities(φ, A, E, B, E1, E2, B1, B2)

    return quantities_fn
