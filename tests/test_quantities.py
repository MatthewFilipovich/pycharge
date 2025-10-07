import jax.numpy as jnp
from scipy.constants import e, epsilon_0, pi

from pycharge import Charge, quantities


def test_stationary_charge():
    charge = Charge(position=lambda t: [0.0, 0.0, 0.0], q=e)
    quantities_fn = quantities([charge])

    x, y, z, t = 1e-9, 0, 0, 0
    r = (x**2 + y**2 + z**2) ** 0.5

    results = quantities_fn(x, y, z, t)

    scalar_expected = (1 / (4 * pi * epsilon_0)) * (e / r)
    scalar_pycharge = results.scalar

    electric_expected = (1 / (4 * pi * epsilon_0)) * (e / r**2) * jnp.array([x, y, z]) / r
    electric_pycharge = results.electric

    assert jnp.allclose(scalar_expected, scalar_pycharge)
    assert jnp.allclose(electric_expected, electric_pycharge)
