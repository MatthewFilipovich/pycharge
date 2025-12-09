"""Type aliases for scalar values and 3D vectors."""

from typing import Sequence, TypeAlias

from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]
