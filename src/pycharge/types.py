"""Type aliases for scalar values and 3D vectors."""

from typing import Sequence, TypeAlias

from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
"""Scalar values: JAX/NumPy arrays or Python floats."""

Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]
"""3D vectors: array-like ``[x, y, z]`` or sequence of three scalars."""
