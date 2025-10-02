import jax.numpy as jnp

from pycharge.sources import dipole_source


def test_dipole_source_structure_and_dynamics():
    d0 = jnp.array([0.0, 0.0, 1e-9])
    q = 1.0
    src = dipole_source(d0=d0, q=q, omega_0=1.0, m=1.0)

    assert hasattr(src, "charges_0")
    assert hasattr(src, "ode_func")
    assert len(src.charges_0) == 2

    charge_a, charge_b = src.charges_0
    assert charge_a.q == q
    assert charge_b.q == -q

    pos_a = jnp.asarray(charge_a.position(0.0))
    pos_b = jnp.asarray(charge_b.position(0.0))
    midpoint = (pos_a + pos_b) / 2
    separation = pos_a - pos_b
    assert jnp.allclose(midpoint, jnp.zeros(3))
    assert jnp.allclose(separation, d0)

    state = jnp.stack((jnp.stack((pos_a, jnp.zeros(3))), jnp.stack((pos_b, jnp.zeros(3)))), axis=0)
    out = src.ode_func(0.0, state, (), None)
    assert out.shape == state.shape
    assert jnp.allclose(out[:, 0], state[:, 1])  # dr/dt == velocity
