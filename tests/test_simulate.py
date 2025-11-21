import jax.numpy as jnp
import pytest
from scipy.constants import e, m_e

from pycharge import dipole_source, free_particle_source, simulate


@pytest.fixture
def dipole1():
    """Defines a dipole source at the origin."""
    return dipole_source(
        d_0=[0.0, 0.0, 1e-11], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 0.0, 0.0]
    )


@pytest.fixture
def dipole2():
    """Defines a dipole source offset on the x-axis."""
    return dipole_source(
        d_0=[0.0, 0.0, 1e-11], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[80e-9, 0.0, 0.0]
    )


@pytest.fixture
def free_particle1():
    """Defines a free particle source with positive charge."""
    return free_particle_source(position_0_fn=lambda t: [1e-9, 0.0, 0.0], q=e, m=m_e)


@pytest.fixture
def free_particle2():
    """Defines a free particle source with negative charge."""
    return free_particle_source(position_0_fn=lambda t: [-1e-9, 0.0, 0.0], q=-e, m=m_e)


@pytest.mark.parametrize(
    "source_names",
    [
        ["dipole1"],
        ["dipole1", "dipole2"],
        ["free_particle1"],
        ["free_particle1", "free_particle2"],
        ["dipole1", "free_particle1"],
    ],
)
def test_simulation(source_names, request):
    sources = [request.getfixturevalue(name) for name in source_names]
    # Simulation time
    t_num = 50
    dt = 1e-18
    ts = jnp.linspace(0, (t_num - 1) * dt, t_num)

    simulate_fn = simulate(sources, ts)
    final_states = simulate_fn()

    assert isinstance(final_states, tuple)
    assert len(final_states) == len(sources)
    for source_state, source in zip(final_states, sources):
        assert source_state.shape == (t_num, len(source.charges_0), 2, 3)
