"""This module defines the Charge class."""

from dataclasses import dataclass
from typing import Callable, Sequence

from jax.typing import ArrayLike
from scipy.constants import e


@dataclass()
class Charge:
    position: Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]]
    q: float = e

    newton_rtol: float = 0.0
    newton_atol: float = 1.48e-8
    root_find_max_steps: int = 256
