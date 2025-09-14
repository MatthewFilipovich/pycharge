"""This module defines the Charge class."""

from typing import Callable, NamedTuple, Sequence, TypeAlias

from jax.typing import ArrayLike
from scipy.constants import e

Scalar: TypeAlias = ArrayLike
Vector3: TypeAlias = ArrayLike | Sequence[ArrayLike]


class Charge(NamedTuple):
    position: Callable[[Scalar], Vector3]
    q: float = e
