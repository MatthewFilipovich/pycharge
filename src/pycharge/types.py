"""Type aliases for PyCharge library.

This module defines type aliases used throughout the PyCharge library for representing
scalar values and 3D vectors in a flexible and type-safe manner.
"""

from typing import Sequence, TypeAlias

from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
"""Type alias for scalar values.

Can be any array-like type compatible with JAX, including Python floats, NumPy scalars,
and JAX arrays.
"""

Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]
"""Type alias for 3D vectors.

Can be either a single array-like object containing three components ``[x, y, z]`` or a
sequence of three array-like objects. Compatible with JAX arrays, NumPy arrays, and
Python sequences.
"""
