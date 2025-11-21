from typing import Sequence, TypeAlias

from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]
