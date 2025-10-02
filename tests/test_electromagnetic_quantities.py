import jax.numpy as jnp
from scipy.constants import e, epsilon_0

from pycharge.charge import Charge
from pycharge.electromagnetic_quantities import (
    electric_field,
    energy_density,
    magnetic_field,
    poynting_vector,
    scalar_potential,
    vector_potential,
)


def test_electromagnetic_quantities_static_charge():
    def pos(t):
        return jnp.array([0.0, 0.0, 0.0])

    ch = Charge(pos, q=e)
    r_mag = 2.0
    point = (jnp.array(0.0), jnp.array(0.0), jnp.array(r_mag), jnp.array(0.0))

    phi = scalar_potential([ch])(*point)
    A = vector_potential([ch])(*point)
    E = electric_field([ch])(*point)
    B = magnetic_field([ch])(*point)
    S = poynting_vector([ch])(*point)
    u = energy_density([ch])(*point)

    coeff = e / (4 * jnp.pi * epsilon_0)
    r_hat = jnp.array([0.0, 0.0, 1.0])

    assert jnp.allclose(phi, coeff / r_mag, rtol=1e-6)
    assert jnp.allclose(A, jnp.zeros(3), atol=1e-12)
    assert jnp.allclose(E, coeff * r_hat / (r_mag**2), rtol=1e-6)
    assert jnp.allclose(B, jnp.zeros(3), atol=1e-12)
    assert jnp.allclose(S, jnp.zeros(3), atol=1e-12)
    expected_energy_density = 0.5 * epsilon_0 * jnp.dot(E, E)
    assert jnp.allclose(u, expected_energy_density, rtol=1e-6)
