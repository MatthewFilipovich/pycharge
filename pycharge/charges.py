#!/usr/bin/env python
"""Module contains the abstract base class `Charge` and example subclasses.

`Charge` is an abstract base class representing a point charge. The class has
abstract methods for the position, velocity, and acceleration of the charge in
the x, y, and z directions.

Subclasses, such as `OscillatingCharge`, define their own time-dependent
trajectories. These classes are used by the `Simulation` class to perform
electromagnetism calculations and simulations.

All units are in SI.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy import constants

# Constants
c = constants.c
e = constants.e


class Charge(ABC):
    """Abstract base class used to define a point charge object.

    Subclasses implement their unique position, velocity, and acceleration
    methods in x, y, and z as functions of time. By default, the velocity and
    acceleration values are determined using finite difference approximations
    from the position methods; however, they can also be explicitly specified.

    Args:
        q (float): Charge value, can be positive or negative.
        h (float): Limit for finite difference calculations for vel and acc.
    """

    def __init__(self, q: float, h: float = 1e-20) -> None:
        self.q = q
        self.h = h

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    @abstractmethod
    def xpos(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return x position of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            x positions at time values.
        """

    @abstractmethod
    def ypos(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return y position of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            y position at time values.
        """

    @abstractmethod
    def zpos(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return z position of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            z position at time values.
        """

    def xvel(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return x velocity of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            x velocity at time values.
        """
        return (self.xpos(t+0.5*self.h)-self.xpos(t-0.5*self.h))/self.h

    def yvel(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return y velocity of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            y velocity at time values.
        """
        return (self.ypos(t+0.5*self.h)-self.ypos(t-0.5*self.h))/self.h

    def zvel(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return z velocity of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            z velocity at time values.
        """
        return (self.zpos(t+0.5*self.h)-self.zpos(t-0.5*self.h))/self.h

    def xacc(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return x acceleration of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            x acceleration at time values.
        """
        return (self.xvel(t+0.5*self.h)-self.xvel(t-0.5*self.h))/self.h

    def yacc(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return y acceleration of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            y acceleration at time values.
        """
        return (self.yvel(t+0.5*self.h)-self.yvel(t-0.5*self.h))/self.h

    def zacc(self, t: Union[ndarray, float]) -> Union[ndarray, float]:
        """Return z acceleration of the charge at specified array of times.

        Args:
            t: 3D meshgrid array of time values.

        Returns:
            z acceleration at time values.
        """
        return (self.zvel(t+0.5*self.h)-self.zvel(t-0.5*self.h))/self.h

    def solve_time(
        self, tr: ndarray, t: ndarray, x: ndarray, y: ndarray, z: ndarray
    ) -> ndarray:
        """Return equation to solve for the retarded time of the charge."""
        tr_larger = tr >= t  # tr must be smaller than t
        tr[tr_larger] = 0  # Avoids pos methods from failing
        # Griffiths Eq. 10.55
        root_equation = ((x-self.xpos(tr))**2 + (y-self.ypos(tr))**2
                         + (z-self.zpos(tr))**2)**0.5 - c*(t-tr)
        root_equation[tr_larger] = np.inf  # Safety for Newton's method
        return root_equation


class StationaryCharge(Charge):
    """Stationary point charge.

    Args:
        position (Tuple[float, float, float]): List of x, y, and z values
            for the stationary position.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self, position: Tuple[float, float, float], q: float = e
    ) -> None:
        super().__init__(q)
        self.position = position

    def xpos(self, t: Union[ndarray, float]) -> float:
        return self.position[0]

    def ypos(self, t: Union[ndarray, float]) -> float:
        return self.position[1]

    def zpos(self, t: Union[ndarray, float]) -> float:
        return self.position[2]

    def xvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def yvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def yacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0


class OscillatingCharge(Charge):
    """Sinusoidally oscillating charge along a specified axis.

    Args:
        origin (Tuple[float, float, float]): List of x, y, and z values for
            the oscillating charge's origin.
        direction (Tuple[float, float, float]): List of x, y, and z values
            for the charge direction vector.
        amplitude (float): Amplitude of the oscillations.
        omega (float): Angular frequency of the oscillations (units: rad/s).
        start_zero (bool): Determines if the charge begins oscillating at t=0.
            Defaults to `False`.
        stop_t (Optional[float]): Time when the charge stops oscillating.
            If `None`, the charge never stops oscillating. Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        amplitude: float,
        omega: float,
        start_zero: bool = False,
        stop_t: Optional[float] = None,
        q: float = e
    ) -> None:
        super().__init__(q)
        self.origin = np.array(origin)
        self.direction = (np.array(direction)
                          / np.linalg.norm(np.array(direction)))
        self.amplitude = amplitude
        self.omega = omega
        self.start_zero = start_zero
        self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        xpos = (self.direction[0]*self.amplitude*np.cos(self.omega*t)
                + self.origin[0])
        if self.start_zero:
            xpos[t < 0] = self.origin[0]
        if self.stop_t is not None:
            xpos[t > self.stop_t] = (self.direction[0]*self.amplitude
                                     * np.cos(self.omega*self.stop_t)
                                     + self.origin[0])
        return xpos

    def ypos(self, t: Union[ndarray, float]) -> ndarray:
        ypos = (self.direction[1]*self.amplitude*np.cos(self.omega*t)
                + self.origin[1])
        if self.start_zero:
            ypos[t < 0] = self.origin[1]
        if self.stop_t is not None:
            ypos[t > self.stop_t] = (self.direction[1]*self.amplitude
                                     * np.cos(self.omega*self.stop_t)
                                     + self.origin[1])
        return ypos

    def zpos(self, t: Union[ndarray, float]) -> ndarray:
        zpos = (self.direction[2]*self.amplitude*np.cos(self.omega*t)
                + self.origin[2])
        if self.start_zero:
            zpos[t < 0] = self.origin[2]
        if self.stop_t is not None:
            zpos[t > self.stop_t] = (self.direction[2]*self.amplitude
                                     * np.cos(self.omega*self.stop_t)
                                     + self.origin[2])
        return zpos

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        xvel = -(self.direction[0]*self.amplitude
                 * self.omega*np.sin(self.omega*t))
        if self.start_zero:
            xvel[t < 0] = 0
        if self.stop_t is not None:
            xvel[t > self.stop_t] = 0
        return xvel

    def yvel(self, t: Union[ndarray, float]) -> ndarray:
        yvel = -(self.direction[1]*self.amplitude
                 * self.omega*np.sin(self.omega*t))
        if self.start_zero:
            yvel[t < 0] = 0
        if self.stop_t is not None:
            yvel[t > self.stop_t] = 0
        return yvel

    def zvel(self, t: Union[ndarray, float]) -> ndarray:
        zvel = -(self.direction[2]*self.amplitude
                 * self.omega*np.sin(self.omega*t))
        if self.start_zero:
            zvel[t < 0] = 0
        if self.stop_t is not None:
            zvel[t > self.stop_t] = 0
        return zvel

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        xacc = -(self.direction[0]*self.amplitude
                 * self.omega**2*np.cos(self.omega*t))
        if self.start_zero:
            xacc[t < 0] = 0
        if self.stop_t is not None:
            xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> ndarray:
        yacc = -(self.direction[1]*self.amplitude
                 * self.omega**2*np.cos(self.omega*t))
        if self.start_zero:
            yacc[t < 0] = 0
        if self.stop_t is not None:
            yacc[t > self.stop_t] = 0
        return yacc

    def zacc(self, t: Union[ndarray, float]) -> ndarray:
        zacc = -(self.direction[2]*self.amplitude
                 * self.omega**2*np.cos(self.omega*t))
        if self.start_zero:
            zacc[t < 0] = 0
        if self.stop_t is not None:
            zacc[t > self.stop_t] = 0
        return zacc


class OrbittingCharge(Charge):
    """Radially orbitting charge in the x-y plane.

    At t=0, the charge is at the position (x=`radius`, y=0, z=0) and orbits
    counter-clockwise.

    Args:
        radius (float): Radius of the orbitting charge trajectory.
        omega (float): Angular frequency of the orbit (units: rad/s).
        start_zero (bool): Determines if the charge begins orbitting at t=0.
            Defaults to `False`.
        stop_t (Optional[float]): Time when the charge stops orbitting.
            If `None`, the charge never stops orbitting. Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        radius: float,
        omega: float,
        start_zero: bool = False,
        stop_t: Optional[float] = None,
        q: float = e
    ) -> None:
        super().__init__(q)
        self.radius = radius
        self.omega = omega
        self.start_zero = start_zero
        self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        xpos = self.radius*np.cos(self.omega*t)
        if self.start_zero:
            xpos[t < 0] = self.radius
        if self.stop_t is not None:
            xpos[t > self.stop_t] = (self.radius
                                     * np.cos(self.omega*self.stop_t))
        return xpos

    def ypos(self, t: Union[ndarray, float]) -> ndarray:
        ypos = self.radius*np.sin(self.omega*t)
        if self.start_zero:
            ypos[t < 0] = 0
        if self.stop_t is not None:
            ypos[t > self.stop_t] = (self.radius
                                     * np.sin(self.omega*self.stop_t))
        return ypos

    def zpos(self, t: Union[ndarray, float]) -> float:
        return 0

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        xvel = -self.radius*self.omega*np.sin(self.omega*t)
        if self.start_zero:
            xvel[t < 0] = 0
        if self.stop_t is not None:
            xvel[t > self.stop_t] = 0
        return xvel

    def yvel(self, t: Union[ndarray, float]) -> ndarray:
        yvel = self.radius*self.omega*np.cos(self.omega*t)
        if self.start_zero:
            yvel[t < 0] = 0
        if self.stop_t is not None:
            yvel[t > self.stop_t] = 0
        return yvel

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        xacc = -self.radius*self.omega**2*np.cos(self.omega*t)
        if self.start_zero:
            xacc[t < 0] = 0
        if self.stop_t is not None:
            xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> ndarray:
        yacc = -self.radius*self.omega**2*np.sin(self.omega*t)
        if self.start_zero:
            yacc[t < 0] = 0
        if self.stop_t is not None:
            yacc[t > self.stop_t] = 0
        return yacc

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0


class LinearAcceleratingCharge(Charge):
    """Linearly accelerating charge along the x-axis.

    At t<0, the charge is stationary at the origin (x=0, y=0, z=0) and
    begins accelerating at t=0. The charge stops accelerating at the time
    given by `stop_t` and continues moving at a constant velocity.

    Args:
        acceleration (float): Accleration along x-axis. Positive value is in +x
            direction, negative is in -x direction.
        stop_t (Optional[float]): Time when charge stops accelerating. If
            `None`, the charge stops accelerating when the velocity reaches
            `0.999c`. Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        acceleration: float,
        stop_t: Optional[float] = None,
        q: float = e
    ) -> None:
        super().__init__(q)
        self.acceleration = acceleration
        if stop_t is None:
            self.stop_t = 0.999*c/acceleration  # Ensures velocity < c
        else:
            self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xpos = 0.5*self.acceleration*t**2
        xpos[t < 0] = 0
        xpos[t > self.stop_t] = (
            0.5*self.acceleration*self.stop_t**2
            + self.acceleration*self.stop_t * (t[t > self.stop_t]-self.stop_t)
        )
        return xpos

    def ypos(self, t: Union[ndarray, float]) -> float:
        return 0

    def zpos(self, t: Union[ndarray, float]) -> float:
        return 0

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xvel = self.acceleration*t
        xvel[t < 0] = 0
        xvel[t > self.stop_t] = self.acceleration*self.stop_t
        return xvel

    def yvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xacc = self.acceleration*np.ones(t.shape)
        xacc[t < 0] = 0
        xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0


class LinearDeceleratingCharge(Charge):
    """Linearly decelerating charge along the x-axis.

    At t<0, the charge is moving at the value of `initial_speed`. At t=0,
    the charge is at the origin (x=0, y=0, z=0) and decelerates until the
    time given by `stop_t` and continues moving at a constant velocity.

    Args:
        deceleration (float): Deceleration along x-axis. Positive value is in
            +x direction, negative is in -x direction.
        initial_speed (float): Initial speed of charge at t<0 along x-axis.
        stop_t (Optional[float]): Time when the charge stops decelerating. If
            `None`, the charge stops decelerating when velocity reaches zero.
            Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        deceleration: float,
        initial_speed: float,
        stop_t: Optional[float] = None,
        q: float = e
    ) -> None:
        super().__init__(q)
        self.deceleration = deceleration
        self.initial_speed = initial_speed
        if stop_t is None:  # Charge stops moving when velocity reaches zero.
            self.stop_t = initial_speed/deceleration
        else:
            self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xpos = self.initial_speed*t
        xpos[t > 0] = (self.initial_speed*t[t > 0]
                       - 0.5*self.deceleration*t[t > 0]**2)
        xpos[t > self.stop_t] = (self.initial_speed * self.stop_t
                                 - 0.5*self.deceleration*self.stop_t**2)
        return xpos

    def ypos(self, t: Union[ndarray, float]) -> float:
        return 0

    def zpos(self, t: Union[ndarray, float]) -> float:
        return 0

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xvel = self.initial_speed*np.ones(t.shape)
        xvel[t > 0] = self.initial_speed - self.deceleration*t[t > 0]
        xvel[t > self.stop_t] = (self.initial_speed
                                 - self.deceleration*self.stop_t)
        return xvel

    def yvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        xacc = np.zeros(t.shape)
        xacc[t > 0] = -self.deceleration
        xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0


class LinearVelocityCharge(Charge):
    """Charge moving with constant velocity along the x-axis.

    At t=0, the charge is at the position given by `init_pos` with the
    magnitude of the velocity given by `speed`.

    Args:
        speed (float): Speed of the charge along the x-axis. Positive value is
            in +x direction, negative is in -x direction.
        init_pos (float): Initial x position at t=0.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(self, speed: float, init_pos: float, q: float = e) -> None:
        super().__init__(q)
        self.speed = speed
        self.init_pos = init_pos

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        t = np.array(t)
        return self.speed*t + self.init_pos

    def ypos(self, t: Union[ndarray, float]) -> float:
        return 0

    def zpos(self, t: Union[ndarray, float]) -> float:
        return 0

    def xvel(self, t: Union[ndarray, float]) -> float:
        return self.speed

    def yvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def yacc(self, t: Union[ndarray, float]) -> float:
        return 0

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0
