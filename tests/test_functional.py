"""This module defines tests for functions."""

import jax
import jax.numpy as jnp
import pytest
from scipy.constants import c

from pycharge.charge import Charge
from pycharge.functional.functional import (
    acceleration,
    interpolate_position,
    position,
    source_time,
    velocity,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_charge():
    """Defines a sample charge with position given by (t, t^2, t^3)."""
    return Charge(q=1.0, position_fn=lambda t: [t, t**2, t**3])


def test_position(sample_charge):
    """Tests the position function."""
    t = 2.0
    expected_pos = jnp.array([2.0, 4.0, 8.0])
    assert jnp.allclose(position(t, sample_charge), expected_pos)


def test_velocity(sample_charge):
    """Tests the velocity function."""
    t = 2.0
    expected_vel = jnp.array([1.0, 4.0, 12.0])
    assert jnp.allclose(velocity(t, sample_charge), expected_vel)


def test_acceleration(sample_charge):
    """Tests the acceleration function."""
    t = 2.0
    expected_acc = jnp.array([0.0, 2.0, 12.0])
    assert jnp.allclose(acceleration(t, sample_charge), expected_acc)


def test_source_time_stationary():
    """Tests the source_time function for a stationary charge."""
    charge = Charge(q=1.0, position_fn=lambda t: jnp.zeros(3))
    r = jnp.array([3.0, 4.0, 0.0])
    t = jnp.array([10.0])

    expected_tr = t - jnp.linalg.norm(r) / c
    tr = source_time(r, t, charge)
    assert jnp.allclose(tr, expected_tr)


@pytest.mark.parametrize("t_val", [0.0, 1e-6, 1e-5, 1e-4, -1e-6, -1e-5, -1e-4])
def test_source_time_moving(t_val):
    v0 = c / 2.0
    charge = Charge(position_fn=lambda tau: [v0 * tau, 0.0, 0.0])

    r = jnp.array([1000.0, 0.0, 0.0])  # observer at x = 1000 m
    R = jnp.linalg.norm(r)
    t_switch = R / v0  # where the physical branch switches
    t = jnp.array(t_val)

    if t_val <= t_switch:  # Branch: R - v0 * tr >= 0  →  tr = (c * t - R) / (c - v0)
        expected_tr = (c * t - R) / (c - v0)
    else:  # Branch: R - v0 * tr < 0   →  tr = (c * t + R) / (c + v0)
        expected_tr = (c * t + R) / (c + v0)

    tr = source_time(r, t, charge)
    print(tr)
    assert jnp.allclose(tr, expected_tr)


def test_interpolate_position():
    """Tests the interpolate_position function."""
    ts = jnp.array([1.0, 2.0, 3.0])
    position_array = jnp.array([[1.0, 1.0, 1.0], [2.0, 4.0, 8.0], [3.0, 9.0, 27.0]])
    velocity_array = jnp.array([[1.0, 2.0, 3.0], [1.0, 4.0, 12.0], [1.0, 6.0, 27.0]])

    def pos_0_fn(t):
        return jnp.array([0.0, 0.0, 0.0]) * t

    pos_fn = interpolate_position(ts, position_array, velocity_array, pos_0_fn)

    # Before start
    t_before = 0.5
    assert jnp.allclose(pos_fn(t_before), pos_0_fn(t_before))

    # After end
    t_after = 3.5
    assert jnp.allclose(pos_fn(t_after), position_array[-1])

    # At knots
    assert jnp.allclose(pos_fn(ts[0]), pos_0_fn(ts[0]))  # interpolate_position uses pos_0_fn for t <= t_start
    assert jnp.allclose(pos_fn(ts[1]), position_array[1])
    assert jnp.allclose(pos_fn(ts[2]), position_array[2])

    # In between
    t_between = 1.5
    # The interpolation is a cubic spline that reproduces the original function
    # for polynomials up to degree 3.
    # With p(t) = (t, t^2, t^3), the interpolated value should match.
    expected_pos = jnp.array([1.5, 1.5**2, 1.5**3])
    assert jnp.allclose(pos_fn(t_between), expected_pos)

    t_between_2 = 2.5
    expected_pos_2 = jnp.array([2.5, 2.5**2, 2.5**3])
    assert jnp.allclose(pos_fn(t_between_2), expected_pos_2)

    # Test with t_end
    pos_fn_tend = interpolate_position(ts, position_array, velocity_array, pos_0_fn, t_end=ts[1])
    t_after_tend = 2.5
    assert jnp.allclose(pos_fn_tend(t_after_tend), position_array[1])
