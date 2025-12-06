"""Point charge representation and solver configuration for electromagnetic simulations."""

from dataclasses import dataclass, field
from typing import Callable

from scipy.constants import e

from pycharge.types import Scalar, Vector3


@dataclass(frozen=True)
class SolverConfig:
    r"""Numerical solver configuration stored in :class:`~pycharge.charge.Charge`.

    Parameters for the two-stage solver in :func:`~pycharge.functional.emission_time`:
    fixed-point iteration followed by Newton refinement.

    Attributes:
        fixed_point_rtol (float): Relative tolerance for fixed-point iteration. Default: ``0.0``.
        fixed_point_atol (float): Absolute tolerance (s). Default: ``1e-20``.
        fixed_point_max_steps (int): Maximum iterations. Default: ``256``.
        fixed_point_throw (bool): Raise exception on failure. Default: ``False``.
        root_find_rtol (float): Relative tolerance for Newton refinement. Default: ``0.0``.
        root_find_atol (float): Absolute tolerance (s). Default: ``1e-20``.
        root_find_max_steps (int): Maximum Newton iterations. Default: ``256``.
        root_find_throw (bool): Raise exception on failure. Default: ``True``.

    Note:
        Default tolerances (``1e-20`` s ≈ ``1e-12`` m) provide sub-atomic
        precision. For faster performance, increase atol to ``1e-18`` or ``1e-16``.
    """

    fixed_point_rtol: float = 0.0
    fixed_point_atol: float = 1e-20
    fixed_point_max_steps: int = 256
    fixed_point_throw: bool = False

    root_find_rtol: float = 0.0
    root_find_atol: float = 1e-20
    root_find_max_steps: int = 256
    root_find_throw: bool = True


@dataclass(frozen=True)
class Charge:
    r"""Point charge with time-dependent trajectory.

    Attributes:
        position_fn (Callable[[Scalar], Vector3]): Position function :math:`\mathbf{r}(t)` mapping
            time (s) to position (m). Must be twice differentiable.
        q (float): Charge (C). Default: ``e`` (elementary charge :math:`\approx 1.602 \times 10^{-19}` C).
        solver_config (SolverConfig): Solver configuration. Default: ``SolverConfig()``.

    Note:
        Velocity :math:`\mathbf{v} = d\mathbf{r}/dt` and acceleration :math:`\mathbf{a} = d^2\mathbf{r}/dt^2`
        computed via JAX autodiff for Liénard-Wiechert calculations.
    """

    position_fn: Callable[[Scalar], Vector3]
    q: float = e
    solver_config: SolverConfig = field(default_factory=SolverConfig)
