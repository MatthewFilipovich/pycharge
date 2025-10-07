"""This module defines the Charge class."""

from typing import Callable, NamedTuple, Sequence

from jax.typing import ArrayLike
from scipy.constants import e


class Charge(NamedTuple):
    position: Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]]
    q: float = e
