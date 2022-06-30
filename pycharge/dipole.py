#!/usr/bin/env python
"""Module contains the `Dipole` class.

The `Dipole` class represents a pair of oscillating point charges whose
trajectories are updated at each time step during the simulation. The positive
and negative charge pair are represented as `_DipoleCharge` objects, which
are a subclass of the `Charge` base class. The trajectories of these charges
are determined at each time step using the `run` method from the instantiated
`Simulation` object, which accepts `Charge` and `Dipole` objects as
initialization parameters.

All units are in SI.
"""
from __future__ import annotations

import hashlib
import types
from collections.abc import Iterable
from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy import constants

from .charges import Charge

# Constants
eps_0 = constants.epsilon_0
c = constants.c
e = constants.e
m_e = constants.m_e


class Dipole():
    r"""Oscillating dipole with a moment that is dependent on driving E field.

    Class simulates an oscillating dipole with a time-dependent origin and
    has a dipole moment that is determined by the Lorentz oscillator equation
    of motion. The dipole moment is updated at each time step in the simulation
    by the `Simulation` object.

    The Lorentz oscillator equation of motion is given by (Novotny Eq. 8.135):

    $d''(t) + \gamma_0*d'(t) + \omega_0^2*d(t) = E_d(t)*q^2/m_{eff}$

    where $E_d(t)$ is the driving electric field (i.e. the component of the
    external electric field along the dipole's axis of polarization).

    Args:
        omega_0 (float): Natural angular frequency of dipole (units: rad/s).
        origin (Union[Tuple[float, float, float], Callable[[float], ndarray]]):
            List of x, y, and z values of dipole's origin (center of mass) or
            function with one input parameter for time that returns a 3 element
            list for x, y, and z values.
        initial_r (Tuple[float, float, float]): List of x, y, and z values
            for the initial displacement vector between the two point charges.
        q (float): Magnitude of the charge value of each point charge. Default
            is `e`.
        m Union[float, Tuple[float, float]]: Mass of the two point charges or
            a 2 element list of the two masses if they are different. Default
            is `m_e`.

    Raises:
        ValueError: Raised if the magnitude of the initial moment is zero.

    Example:
        Below is an origin function that oscillates along the x-axis with an
        amplitude of `1e-10 m` and angular frequency of `1e12*2*pi rad/s`:

        ```python
        def fun_origin(t):
            return np.array((1e-10*np.cos(1e12*2*np.pi*t), 0, 0))
        ```
    """

    def __init__(
            self,
            omega_0: float,
            origin: Union[Tuple, Callable[[float], ndarray]],
            initial_r: Tuple[float, float, float],
            q: float = e,
            m: Union[float, Tuple[float, float]] = m_e,
    ) -> None:
        self.q = q
        # The mass parameter is either a list-like object or a float
        if isinstance(m, (ndarray, Iterable)):
            self.m = m
        else:
            self.m = (m, m)
        self.m_eff = self.m[0]*self.m[1]/(self.m[0]+self.m[1])
        self.omega_0 = omega_0
        # The origin parameter is either a list-like object or a function
        if isinstance(origin, (ndarray, Iterable)):
            self.origin = self._origin_fun(np.array(origin))  # Function origin
        else:  # Create a deep copy of the passed function
            self.origin = types.FunctionType(
                origin.__code__, origin.__globals__, origin.__name__,
                origin.__defaults__, origin.__closure__
            )
        self.id = hashlib.sha1(str.encode(str(self.origin(1e-16))+str(omega_0)
                                          + str(q)+str(m) + str(initial_r))
                               ).hexdigest()  # Object id
        if np.linalg.norm(initial_r) == 0:
            raise ValueError('Initial moment must not be zero.')
        self.initial_r = np.array(initial_r)
        self.polar_dir = abs(self.initial_r / np.linalg.norm(self.initial_r))
        self.gamma_0 = (1/(4*np.pi*eps_0)*2*self.q**2*self.omega_0**2
                        / (3*self.m_eff*c**3))
        self.charge_pair = (
            self._DipoleCharge(self, True),
            self._DipoleCharge(self, False)
        )
        # Attributes below are defined in the `reset` method
        self.t_index = None  # Stores the current time step during simulation
        self.dt = None
        self.moment_disp = None
        self.moment_vel = None
        self.moment_acc = None
        self.E_total = None
        self.E_vel = None
        self.E_acc = None

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.id == other.id

    def update_timestep(
        self,
        moment_disp: ndarray,
        moment_vel: ndarray,
        moment_acc: ndarray,
        E_driving: Union[None, Tuple[ndarray, ndarray, ndarray]]
    ) -> None:
        """Update the array attributes at each time step."""
        self.t_index += 1
        self.moment_disp[:, self.t_index] = moment_disp
        self.moment_vel[:, self.t_index] = moment_vel
        self.moment_acc[:, self.t_index] = moment_acc
        if E_driving is not None:
            self.E_total[:, self.t_index] = E_driving[0]
            self.E_vel[:, self.t_index] = E_driving[1]
            self.E_acc[:, self.t_index] = E_driving[2]
        t = self.dt*self.t_index
        h = 1e-21  # Limit used to calcuate derivatives of the origin position
        origin_vel = (self.origin(t+h)-self.origin(t-h))/(2*h)
        origin_acc = (self.origin(t+h)-2*self.origin(t)+self.origin(t-h))/h**2
        m_frac = self.m[1]/(self.m[0]+self.m[1])  # Determine COM
        self.charge_pair[0].update_timestep(  # Update first charge
            self.origin(t)+moment_disp*m_frac,
            origin_vel+moment_vel*m_frac, origin_acc+moment_acc*m_frac
        )
        self.charge_pair[1].update_timestep(  # Update second charge
            self.origin(t)-moment_disp*(1-m_frac),
            origin_vel-moment_vel*(1-m_frac), origin_acc-moment_acc*(1-m_frac)
        )

    def get_origin_position(self, magnitude: bool = False) -> ndarray:
        """Return the position of the dipole's origin at each time step.

        Args:
            magnitude: Return the magnitude of the origin position instead of a
                3D vector if `True`. Defaults to `False`.

        Returns:
            Origin at each time step, either the magnitude of the position (1D
            array of size N) or the position vector (2D array of size 3 x N).

        """
        origin_position = np.zeros((3, self.t_index+1))
        for i in np.arange(self.t_index+1):
            origin_position[:, i] = self.origin(self.dt*i)
        if magnitude:
            return np.linalg.norm(origin_position, axis=0)
        return origin_position

    def get_kinetic_energy(self, exclude_origin: bool = True) -> ndarray:
        """Return the kinetic energy of the dipole at each time step.

        The kinetic energy of just the dipole moment can be determined by
        excluding the kinetic energy from the origin's movement.

        Args:
            exclude_origin: Kinetic energy calculation excludes the movement of
                the dipole's origin. Defaults to `True`.

        Returns:
            Kinetic energy at each time step.
        """
        if exclude_origin:
            return 0.5*self.m_eff*np.linalg.norm(self.moment_vel, axis=0)**2
        charge_KE = (0.5*self.m_eff *
                     np.linalg.norm(self.moment_vel, axis=0)**2)
        return charge_KE  # Double KE since there are two charges

    def get_E_driving(self, field_type: str = 'Total') -> ndarray:
        """Return the magnitude of the driving electric field.

        The driving electric field is the component of the external electric
        field experienced by the charge along the direction of polarization.
        The returned field type (`Total`, `Velocity`, or `Acceleration`) can
        be specified.

        Args:
            field_type: Return either the `Total`, `Velocity`, or
                `Acceleration` field. Defaults to `Total`.

        Raises:
            ValueError: Input for `field_type` argument is invalid.

        Returns:
            Magnitude of the driving electric field at each time step.
        """
        if field_type == 'Total':
            return np.linalg.norm(self.E_total, axis=0)
        if field_type == 'Velocity':
            return np.linalg.norm(self.E_vel, axis=0)
        if field_type == 'Acceleration':
            return np.linalg.norm(self.E_acc, axis=0)
        raise ValueError('Invalid field')

    def reset(self, timesteps: float, dt: float, save_E: bool) -> None:
        """Initialize the moment and E arrays, and `_DipoleCharge` pair."""
        self.t_index = 0
        self.dt = dt
        self.moment_disp = np.ones((3, timesteps))*np.inf
        self.moment_vel = np.ones((3, timesteps))*np.inf
        self.moment_acc = np.ones((3, timesteps))*np.inf
        self.moment_disp[:, 0] = self.initial_r
        self.moment_vel[:, 0] = 0
        self.moment_acc[:, 0] = 0
        if save_E:
            self.E_total = np.ones((3, timesteps))*np.inf
            self.E_vel = np.ones((3, timesteps))*np.inf
            self.E_acc = np.ones((3, timesteps))*np.inf
        for charge in self.charge_pair:
            charge.t_index = 0
            charge.dt = dt
            charge.position = np.ones((3, timesteps))*np.inf
            charge.velocity = np.ones((3, timesteps))*np.inf
            charge.acceleration = np.ones((3, timesteps))*np.inf
            m_frac = self.m[1]/(self.m[0]+self.m[1])  # Determine COM
            if charge.positive_charge:  # Set initial position
                charge.position[:, 0] = self.origin(0) + self.initial_r*m_frac
            else:
                charge.position[:, 0] = (self.origin(0)
                                         - self.initial_r*(1-m_frac))
            charge.velocity[:, 0] = 0
            charge.acceleration[:, 0] = 0

    def _origin_fun(self, origin: ndarray) -> Callable[[float], ndarray]:
        """Return a function for a stationary origin."""
        def stationary_origin(t):  # pylint: disable=unused-argument
            return np.array(origin)
        return stationary_origin

    class _DipoleCharge(Charge):
        """Positive or negative point charge in the oscillating dipole.

        Position, velocity, and acceleration properties are updated at each
        time step during the simulation.

        Args:
            dipole (Dipole): Instantiated `Dipole` object.
            positive_charge (bool): Positive charge if `True`, else negative.
            timesteps (float): Number of time steps in the simulation.
        """

        def __init__(
            self,
            dipole: Dipole,
            positive_charge: bool,
        ) -> None:
            self.positive_charge = positive_charge
            if positive_charge:
                super().__init__(dipole.q)
            else:
                super().__init__(-dipole.q)
            self.id = hashlib.sha1(str.encode(dipole.id+str(positive_charge)
                                              )).hexdigest()  # Object id
            # Attributes below are defined by `Dipole` in `reset` method
            self.t_index = None
            self.dt = None
            self.position = None
            self.velocity = None
            self.acceleration = None

        def __eq__(self, other: Any):
            return isinstance(other, self.__class__) and self.id == other.id

        def update_timestep(
            self,
            position: ndarray,
            velocity: ndarray,
            acceleration: ndarray
        ) -> None:
            """Update trajectory of charge at each timestep.

            Method used by `Simulation` object in the `run` method.
            """
            self.t_index += 1
            self.position[:, self.t_index] = position
            self.velocity[:, self.t_index] = velocity
            self.acceleration[:, self.t_index] = acceleration

        def xpos(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.position[0])

        def ypos(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.position[1])

        def zpos(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.position[2])

        def xvel(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.velocity[0])

        def yvel(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.velocity[1])

        def zvel(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.velocity[2])

        def xacc(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.acceleration[0])

        def yacc(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.acceleration[1])

        def zacc(self, t: Union[ndarray, float]) -> ndarray:
            return self._get_array_values(np.copy(t), self.acceleration[2])

        def _get_array_values(
            self,
            t: ndarray,
            array: ndarray
        ) -> ndarray:
            """Perform linear interpolation from array values."""
            t /= self.dt  # scale t
            t_min = t.astype(int)
            t_min[t_min < 0] = 0
            ret_arr = (array[t_min+1]-array[t_min])*(t-t_min) + array[t_min]
            ret_arr[t < 0] = array[0]  # Dipole is stationary for t<0
            return ret_arr

        def solve_time(
            self, tr: ndarray, t: ndarray, x: ndarray, y: ndarray, z: ndarray
        ) -> ndarray:
            """Return equation to solve for the retarded time of the charge.

            Since the position of the dipole's charges are determined
            dynamically at each time step, this method returns inf if the guess
            by Newton's method is larger than t."""
            tr_larger = tr >= t  # tr must be smaller than t
            tr[tr_larger] = 0  # Avoids pos methods from failing
            # Griffiths Eq. 10.55
            root_equation = ((x-self.xpos(tr))**2 + (y-self.ypos(tr))**2
                             + (z-self.zpos(tr))**2)**0.5 - c*(t-tr)
            root_equation[tr_larger] = np.inf  # Safety for Newton's method
            return root_equation
