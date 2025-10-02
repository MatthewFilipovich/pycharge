import jax.numpy as jnp
from scipy.constants import e as elementary_charge

from pycharge import Charge


def test_charge_initialization():
    def trajectory(t):
        return jnp.array([0.0, 0.0, 0.0])

    charge = Charge(trajectory)
    assert charge.q == elementary_charge
    assert jnp.allclose(jnp.asarray(charge.position(0.0)), jnp.zeros(3))


def test_charge_custom_charge():
    def trajectory(t):
        return jnp.array([t, 0.0, 0.0])

    q = 2.0 * elementary_charge
    charge = Charge(trajectory, q=q)
    assert charge.q == q
