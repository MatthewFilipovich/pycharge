"""Point charge representation for electromagnetic simulations.

This module defines the Charge class, which represents a point charge with
time-dependent position for use in electromagnetic field calculations.
"""

from dataclasses import dataclass
from typing import Callable

from scipy.constants import e

from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class Charge:
    r"""Represents a point charge with time-dependent position.

    This class encapsulates a point charge with its charge magnitude and a function
    describing its position over time. It also includes configurable parameters for
    the numerical solvers used to compute retarded times in electromagnetic calculations.

    Attributes
    ----------
    position_fn : Callable[[Scalar], Vector3]
        A callable that takes time :math:`t` and returns the 3D position vector
        :math:`\mathbf{r}(t) = [x(t), y(t), z(t)]` of the charge.
    q : float, optional
        Charge magnitude in Coulombs. Defaults to elementary charge
        :math:`e \approx 1.602 \times 10^{-19}` C.
    fixed_point_rtol : float, optional
        Relative tolerance for fixed-point iteration solver. Default is 0.0.
    fixed_point_atol : float, optional
        Absolute tolerance for fixed-point iteration solver. Default is :math:`10^{-20}`.
    fixed_point_max_steps : int, optional
        Maximum iterations for fixed-point solver. Default is 256.
    fixed_point_throw : bool, optional
        Whether to raise exception if fixed-point solver doesn't converge. Default is False.
    root_find_rtol : float, optional
        Relative tolerance for Newton's method root finder. Default is 0.0.
    root_find_atol : float, optional
        Absolute tolerance for Newton's method root finder. Default is :math:`10^{-20}`.
    root_find_max_steps : int, optional
        Maximum iterations for root finding. Default is 256.
    root_find_throw : bool, optional
        Whether to raise exception if root finder doesn't converge. Default is True.

    Notes
    -----
    The solver parameters control the accuracy of retarded time calculations,
    which are critical for computing electromagnetic fields from moving charges.
    The default values work well for most applications, but can be adjusted for
    specific accuracy or performance requirements.
    """

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
