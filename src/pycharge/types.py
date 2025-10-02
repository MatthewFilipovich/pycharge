from typing import Callable, Literal, Sequence, TypeAlias

from jax import Array
from jax.typing import ArrayLike

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]

PositionFn: TypeAlias = Callable[[Scalar], Vector3]
SpaceTimeFn: TypeAlias = Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]
FieldComponent: TypeAlias = Literal["total", "velocity", "acceleration"]
