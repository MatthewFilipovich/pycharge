from dataclasses import dataclass
from typing import Callable

from scipy.constants import e

from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class Charge:
    position_fn: Callable[[Scalar], Vector3]
    q: float = e

    fixed_point_rtol: float = 0.0
    fixed_point_atol: float = 1e-20
    fixed_point_max_steps: int = 256
    fixed_point_throw: bool = False

    root_find_rtol: float = 0.0
    root_find_atol: float = 1e-20
    root_find_max_steps: int = 256
    root_find_throw: bool = True
