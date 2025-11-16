"""This module defines the Charge class."""

from dataclasses import dataclass
from typing import Callable, Sequence

from jax.typing import ArrayLike
from scipy.constants import e


@dataclass()
class Charge:
    """Represents a single point charge.

    This dataclass holds the physical properties of a point charge and the
    parameters for the root-finding algorithm used to calculate its
    retarded time.

    Parameters
    ----------
    position : Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]]
        A time-dependent function that returns the (x, y, z) coordinates
        of the charge. The velocity and acceleration are automatically
        derived from this function using JAX's automatic differentiation.
    q : float, optional
        The electric charge of the particle in Coulombs. Defaults to the
        elementary charge, `e`.
    newton_rtol : float, optional
        The relative tolerance for the Newton root-finding solver used for
        retarded time calculations. Defaults to 0.0.
    newton_atol : float, optional
        The absolute tolerance for the Newton root-finding solver.
        Defaults to 1.48e-8.
    root_find_max_steps : int, optional
        The maximum number of iterations for the root-finding solver.
        Defaults to 256.

    """

    position: Callable[[ArrayLike], ArrayLike | Sequence[ArrayLike]]
    q: float = e

    fixed_point_rtol: float = 0.0
    fixed_point_atol: float = 1e-20
    fixed_point_max_steps: int = 256
    fixed_point_throw: bool = False

    root_find_rtol: float = 0.0
    root_find_atol: float = 1e-20
    root_find_max_steps: int = 256
    root_find_throw: bool = True
