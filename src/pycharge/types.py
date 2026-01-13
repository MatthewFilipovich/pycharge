"""Type aliases for scalar values and 3D vectors."""

from collections.abc import Sequence
from typing import TypeAlias

from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]
