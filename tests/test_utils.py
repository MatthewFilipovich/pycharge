import jax.numpy as jnp

from pycharge.utils import cross_1d, dot_1d, interpolate_position


def test_interpolate_position_midpoint():
    # Create a simple linear motion with known velocity
    ts = jnp.array([0.0, 1.0, 2.0])

    def pos0(t):
        return jnp.array([0.0, 0.0, 0.0])

    # position_array corresponds to positions at ts for a particle moving at unit speed along z
    position_array = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    velocity_array = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    interp = interpolate_position(ts, pos0, position_array, velocity_array)

    # midpoint at t=0.5 should be near z=0.5
    p = interp(0.5)
    assert p.shape == (3,)
    assert jnp.allclose(p, jnp.array([0.0, 0.0, 0.5]), atol=1e-6)


def test_interpolate_position_respects_bounds_and_default():
    ts = jnp.array([0.0, 1.0])

    def pos0(t):
        return jnp.array([1.0, 2.0, 3.0])

    position_array = jnp.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    velocity_array = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    default = jnp.array([-1.0, -1.0, -1.0])

    interp_default = interpolate_position(ts, pos0, position_array, velocity_array, default)
    assert jnp.allclose(jnp.asarray(interp_default(-0.5)), pos0(-0.5))
    assert jnp.allclose(jnp.asarray(interp_default(1.5)), default)

    interp_nan = interpolate_position(ts, pos0, position_array, velocity_array)
    after = jnp.asarray(interp_nan(1.5))
    assert jnp.all(jnp.isnan(after))


def test_cross_and_dot_vectorized():
    a = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    cross = cross_1d(a, b)
    dot = dot_1d(a, b)

    expected_cross = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    expected_dot = jnp.array([0.0, 0.0])

    assert jnp.allclose(cross, expected_cross)
    assert jnp.allclose(dot, expected_dot)
