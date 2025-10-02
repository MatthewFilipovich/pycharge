import jax.numpy as jnp
from scipy.constants import e, epsilon_0

from pycharge.charge import Charge
from pycharge.electromagnetic_quantities import scalar_potential


def test_scalar_potential_static_charge():
    # static charge at origin
    def position(t):
        return jnp.array([0.0, 0.0, 0.0])

    charge = Charge(position, q=e)
    phi_fn = scalar_potential([charge])

clear    # evaluate at r = (0,0,2), t=0
    val = phi_fn(jnp.array(0.0), jnp.array(0.0), jnp.array(2.0), jnp.array(0.0))
    expected = e / (4 * jnp.pi * epsilon_0 * 2.0)
    assert jnp.allclose(val, expected, rtol=1e-6, atol=0.0)


def test_accepts_generator_of_charges():
    def position(t):
        return jnp.array([0.0, 0.0, 0.0])

    def gen():
        yield Charge(position, q=e)

    phi_fn = scalar_potential(gen())
    val = phi_fn(jnp.array(1.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    assert jnp.isfinite(val)


def test_scalar_potential_vectorized_inputs():
    def position(t):
        return jnp.array([0.0, 0.0, 0.0])

    charge = Charge(position, q=e)
    phi_fn = scalar_potential([charge])

    z = jnp.array([1.0, 2.0, 4.0])
    values = phi_fn(jnp.zeros_like(z), jnp.zeros_like(z), z, jnp.zeros_like(z))
    expected = e / (4 * jnp.pi * epsilon_0) / z
    assert jnp.allclose(values, expected, rtol=1e-6)
