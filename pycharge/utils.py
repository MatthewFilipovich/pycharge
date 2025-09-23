"""This module defines utility functions."""

import jax
import jax.numpy as jnp
from jax import Array


def cross_1d(a: Array, b: Array) -> Array:
    """Computes the cross product of two vectors."""
    a_flat, b_flat = a.reshape(-1, 3), b.reshape(-1, 3)  # Flatten for vmap
    out_shape = a.shape  # Get the output shape
    return jax.vmap(jnp.cross)(a_flat, b_flat).reshape(out_shape)


def dot_1d(a: Array, b: Array) -> Array:
    """Computes the dot product of two vectors."""
    a_flat, b_flat = a.reshape(-1, 3), b.reshape(-1, 3)  # Flatten for vmap
    out_shape = a.shape[:-1]  # Get the output shape
    return jax.vmap(jnp.dot)(a_flat, b_flat).reshape(out_shape)


def make_cubic_hermite_spline(x0, y0, m0, x1, y1, m1):
    """
    Creates a callable function for cubic Hermite spline interpolation.

    Args:
        x0, y0: Position of the first point.
        m0: Slope (first derivative) at the first point.
        x1, y1: Position of the second point.
        m1: Slope (first derivative) at the second point.

    Returns:
        A function that takes an x-value and returns the interpolated y-value.
    """
    # Calculate deltas
    dx = x1 - x0
    dy = y1 - y0

    # Calculate the coefficients for the normalized polynomial P(t) = at^3 + bt^2 + ct + d
    d = y0
    c = m0 * dx
    b = 3 * dy - (2 * m0 + m1) * dx
    a = (m0 + m1) * dx - 2 * dy

    def interpolator(x):
        """
        Calculates the interpolated y-value for a given x.
        This inner function has access to a, b, c, d, x0, and x1.
        """
        # if x < x0 or x > x1:
        #     raise ValueError(f"Input x={x} is outside the interpolation interval [{x0}, {x1}]")

        # Normalize x to the [0, 1] interval as 't'
        t = (x - x0) / dx

        # Evaluate the cubic polynomial
        return a * t**3 + b * t**2 + c * t + d

    return interpolator
