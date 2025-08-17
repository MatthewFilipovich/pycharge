from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias

from jax.typing import ArrayLike
from scipy.constants import e

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]


@dataclass(frozen=True)
class Charge:
    position: Callable[[Scalar], Vector3]
    q: float = e
