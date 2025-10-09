import jax.numpy as jnp
from scipy.constants import e, epsilon_0, pi

from pycharge import Charge, potentials_and_fields


def test_stationary_charge():
    charge = Charge(position=lambda t: [0.0, 0.0, 0.0], q=e)
    potentials_and_fields_fn = potentials_and_fields([charge])

    x, y, z, t = 1e-9, 0, 0, 0
    r = (x**2 + y**2 + z**2) ** 0.5

    quantities = potentials_and_fields_fn(x, y, z, t)

    scalar_expected = (1 / (4 * pi * epsilon_0)) * (e / r)
    scalar_pycharge = quantities.scalar

    electric_expected = (1 / (4 * pi * epsilon_0)) * (e / r**2) * jnp.array([x, y, z]) / r
    electric_pycharge = quantities.electric

    assert jnp.allclose(scalar_expected, scalar_pycharge)
    assert jnp.allclose(electric_expected, electric_pycharge)


# TODO: Ensure I compare v1 and v2 fields and potentials to check for any errors!
