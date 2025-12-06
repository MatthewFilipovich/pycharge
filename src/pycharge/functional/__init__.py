"""Core functional utilities for charge dynamics and electromagnetic calculations.

This submodule exports essential functions for working with charge trajectories,
including position, velocity, acceleration calculations, retarded time solvers,
and trajectory interpolation utilities.

Functions
---------
interpolate_position
    Interpolate 3D position using cubic splines from discrete trajectory data
position
    Compute position components from a time-dependent trajectory function
velocity
    Compute velocity vector by differentiating trajectory
acceleration
    Compute acceleration vector by differentiating velocity
source_time
    Find retarded time that satisfies field propagation constraints

Notes
-----
All functions use JAX for automatic differentiation and can be JIT-compiled for
performance. The trajectory functions operate on 3D position arrays with components
:math:`\mathbf{r}(t) = [x(t), y(t), z(t)]`.
"""

from .functional import acceleration, interpolate_position, position, source_time, velocity

__all__ = [
    "acceleration",
    "interpolate_position",
    "position",
    "source_time",
    "velocity",
]
