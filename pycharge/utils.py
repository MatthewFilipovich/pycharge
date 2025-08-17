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
