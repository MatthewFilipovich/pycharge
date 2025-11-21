import jax
import jax.numpy as jnp
import pytest
from scipy.constants import e, epsilon_0, mu_0, pi

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("x, y, z", [(1e-9, -1e-9, 1.2), (0.5e-9, 0.1, 0.5), (-0.8e-9, -0.32, 5.0)])
def test_stationary_charge(x, y, z):
    """Test quantities for a single stationary charge against Coulomb's law."""
    charge_pos = jnp.array([0.1, -0.2, 0.3])  # Charge not at origin
    charge = Charge(lambda t: charge_pos, e)
    potentials_and_fields_fn = potentials_and_fields([charge])

    t = 0.0
    # Calculate observation position
    r_vec = jnp.array([x, y, z])
    r_mag = jnp.linalg.norm(r_vec)

    quantities = potentials_and_fields_fn(x, y, z, t)

    # Expected values for a stationary charge (Coulomb's Law)
    scalar_expected = (1 / (4 * pi * epsilon_0)) * (e / r_mag)
    vector_expected = jnp.zeros(3)
    electric_expected = (1 / (4 * pi * epsilon_0)) * (e / r_mag**2) * (r_vec / r_mag)
    magnetic_expected = jnp.zeros(3)

    # Check total quantities
    assert jnp.allclose(quantities.scalar, scalar_expected)
    assert jnp.allclose(quantities.vector, vector_expected)
    assert jnp.allclose(quantities.electric, electric_expected)
    assert jnp.allclose(quantities.magnetic, magnetic_expected)

    # Check field components
    # For a stationary charge, E-field is purely Coulombic (term1), and B-field is zero.
    assert jnp.allclose(quantities.electric_term1, electric_expected)
    assert jnp.allclose(quantities.electric_term2, jnp.zeros(3))
    assert jnp.allclose(quantities.magnetic_term1, jnp.zeros(3))
    assert jnp.allclose(quantities.magnetic_term2, jnp.zeros(3))


@pytest.mark.parametrize("z_obs", [0.0, 1e-4, 1e-3])
def test_current_loop_b_field(z_obs):
    """
    Test B-field on the axis of a current loop.
    This test models a current loop with many discrete charges and compares
    the on-axis B-field with the analytical Biot-Savart law solution.
    """
    # Simulation parameters from the example
    num_charges = 3
    radius = 1e-3
    omega = 1e6
    q = e

    def position(t):
        x = radius * jnp.cos(omega * t)
        y = radius * jnp.sin(omega * t)
        return jnp.array([x, y, 0.0])

    charges = [Charge(position_fn=position, q=q) for _ in range(num_charges)]

    potentials_and_fields_fn = potentials_and_fields(charges)

    # Observation at (0, 0, z_obs) at t=0
    quantities = potentials_and_fields_fn(0.0, 0.0, z_obs, 0.0)

    # Expected B-field from Biot-Savart Law
    current = num_charges * q * omega / (2 * jnp.pi)
    b_mag_expected = (mu_0 * current * radius**2) / (2 * (z_obs**2 + radius**2) ** 1.5)
    b_vec_expected = jnp.array([0.0, 0.0, b_mag_expected])

    assert jnp.allclose(quantities.magnetic, b_vec_expected)


def test_vectorized_inputs():
    """Test that potentials_and_fields can handle vectorized (array) inputs."""
    charge = Charge(position_fn=lambda t: jnp.array([0.0, 0.0, 0.0]), q=e)
    potentials_and_fields_fn = potentials_and_fields([charge])
    dim_shape = (3, 4, 5)

    x = jnp.zeros(dim_shape) + 1e-9
    y = jnp.zeros(dim_shape)
    z = jnp.zeros(dim_shape)
    t = jnp.zeros(dim_shape)

    quantities = potentials_and_fields_fn(x, y, z, t)

    assert quantities.scalar.shape == dim_shape
    assert quantities.vector.shape == (*dim_shape, 3)
    assert quantities.electric.shape == (*dim_shape, 3)
    assert quantities.magnetic.shape == (*dim_shape, 3)


def test_jit_compatibility():
    """Test that potentials_and_fields function is compatible with JAX JIT compilation."""
    charge = Charge(position_fn=lambda t: jnp.array([0.0, 0.0, 0.0]), q=e)
    potentials_and_fields_fn = potentials_and_fields([charge])

    @jax.jit
    def compute_quantities(x, y, z, t):
        return potentials_and_fields_fn(x, y, z, t)

    x = jnp.array([1e-9, 2e-9])
    y = jnp.array([0.0, 0.0])
    z = jnp.array([0.0, 0.0])
    t = jnp.array([0.0, 0.0])

    quantities = compute_quantities(x, y, z, t)

    assert quantities.scalar.shape == (2,)
    assert quantities.vector.shape == (2, 3)
    assert quantities.electric.shape == (2, 3)
    assert quantities.magnetic.shape == (2, 3)


def test_zero_charges_raises_error():
    """Test that potentials_and_fields handles zero charges without error."""
    with pytest.raises(ValueError):
        potentials_and_fields([])


def test_xyzt_shape_mismatch_raises_error():
    """Test that potentials_and_fields raises an error for shape mismatch in inputs."""
    charge = Charge(position_fn=lambda t: jnp.array([0.0, 0.0, 0.0]), q=e)
    potentials_and_fields_fn = potentials_and_fields([charge])

    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0])
    z = jnp.array([0.0, 1.0])
    t = jnp.array([0.0, 1.0])

    with pytest.raises(ValueError):
        potentials_and_fields_fn(x, y, z, t)
