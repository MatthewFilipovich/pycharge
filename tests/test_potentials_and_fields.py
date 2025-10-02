import jax.numpy as jnp
from scipy.constants import e, epsilon_0

from pycharge.charge import Charge
from pycharge.config import Config
from pycharge.potentials_and_fields import potentials_and_fields


def test_potentials_scalar_static_charge_matches_coulomb():
    def pos(t):
        return jnp.array([0.0, 0.0, 0.0])

    ch = Charge(pos, q=e)
    fn = potentials_and_fields([ch], scalar=True)
    phi = fn(jnp.array(1.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))["scalar"]
    expected = e / (4 * jnp.pi * epsilon_0 * 1.0)
    assert jnp.allclose(phi, expected, rtol=1e-6)


def test_potentials_input_shape_mismatch_raises():
    def pos(t):
        return jnp.array([0.0, 0.0, 0.0])

    ch = Charge(pos, q=e)
    fn = potentials_and_fields([ch], scalar=True)
    try:
        _ = fn(jnp.array([1.0, 2.0]), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_potentials_full_fields_static_charge():
    def pos(t):
        return jnp.array([0.0, 0.0, 0.0])

    ch = Charge(pos, q=e)
    fn = potentials_and_fields([ch], scalar=True, vector=True, electric=True, magnetic=True)
    res = fn(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))

    r_mag = 1.0
    r_hat = jnp.array([0.0, 0.0, 1.0])
    coeff = e / (4 * jnp.pi * epsilon_0)

    assert jnp.allclose(res["scalar"], coeff / r_mag, rtol=1e-6)
    assert jnp.allclose(res["vector"], jnp.zeros(3), atol=1e-12)
    assert jnp.allclose(res["electric"], coeff * r_hat / (r_mag**2), rtol=1e-6)
    assert jnp.allclose(res["magnetic"], jnp.zeros(3), atol=1e-12)


def test_potentials_velocity_component_matches_total_for_static_charge():
    def pos(t):
        return jnp.array([0.0, 0.0, 0.0])

    ch = Charge(pos, q=e)
    fn_total = potentials_and_fields([ch], electric=True)
    fn_velocity = potentials_and_fields([ch], electric=True, config=Config(field_component="velocity"))

    total_E = fn_total(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))["electric"]
    velocity_E = fn_velocity(jnp.array(0.0), jnp.array(0.0), jnp.array(1.0), jnp.array(0.0))["electric"]

    assert jnp.allclose(total_E, velocity_E, rtol=1e-6)
