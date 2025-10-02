import jax.numpy as jnp

from pycharge.charge import Charge
from pycharge.simulate import rk4_step, simulate
from pycharge.sources import Source


def test_simulate_returns_function_and_static_state():
    def pos(t):
        return jnp.array([1.0, 0.0, 0.0])

    ch = Charge(pos)

    def zero_force_ode(_t, state, _other_charges, _config):
        velocities = state[:, 1]
        zeros = jnp.zeros_like(velocities)
        return jnp.stack((velocities, zeros), axis=1)

    src = Source(charges_0=(ch,), ode_func=zero_force_ode)
    sim_fn = simulate([src])
    assert callable(sim_fn)

    ts = jnp.linspace(0.0, 1.0, 3)
    states = sim_fn(ts)
    assert len(states) == 1
    state = states[0]
    assert state.shape == (ts.shape[0], 1, 2, 3)
    positions = state[:, 0, 0]
    velocities = state[:, 0, 1]
    assert jnp.allclose(positions, jnp.array([[1.0, 0.0, 0.0]] * ts.shape[0]))
    assert jnp.allclose(velocities, jnp.zeros((ts.shape[0], 3)))


def test_rk4_step_constant_derivative():
    def derivative(_t, y, _other, _config):
        return jnp.ones_like(y)

    y0 = jnp.array([0.0, 0.0])
    out = rk4_step(derivative, 0.0, 1.0, y0, 1.0, (), None)
    assert jnp.allclose(out, jnp.array([1.0, 1.0]))
