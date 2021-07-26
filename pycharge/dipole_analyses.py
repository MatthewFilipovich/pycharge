#!/usr/bin/env python
"""Module contains dipole analysis functions for use in `pycharge`."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit

from pycharge.dipole import Dipole

# Constants
c = constants.c
eps_0 = constants.epsilon_0
hbar = constants.hbar


def calculate_dipole_properties(
    dipole: Dipole,
    first_index: int,
    plot: bool = False,
    print_values: bool = False
) -> Tuple[float, float]:
    r"""Return the dipole's decay rate and frequency shift from kinetic energy.

    The function calculates the modified decay rate (\gamma) and frequency
    shift (\delta_12) of the dipole by fitting these variables to the
    kinetic energy at each time step. The first index of the kinetic energy
    array used to calculate these parameters can be specified (calculations
    should only use performed using kinetic energy values after the dipoles in
    the simulation reach equilibrium.)

    Args:
        dipole: `Dipole` object for calculating properties.
        first_index: The first index of the kinetic energy array that is
            used to calculate the dipole properties.
        plot: Plots the kinetic energy fit if `True`. Defaults to `False`.
        print_values: Plots \gamma_12 and \delta_12 if `True`. Defaults to
        `False`.

    Returns:
        \delta_12 and \gamma scaled by \gamma_0.
    """
    def decaying_fun(t, a, gamma, omega, phi):  # Function to fit KE.
        return a * np.exp(-gamma*t)*np.sin(omega*t+phi)**2

    kinetic_energy = dipole.get_kinetic_energy()[first_index:]
    gamma_0 = dipole.gamma_0
    omega_0 = dipole.omega_0
    t_array = dipole.dt*np.arange(first_index,
                                  first_index+len(kinetic_energy))

    # Initial curve fit guess for parameters use \gamma_0 and \omega_0.
    # pylint: disable=unbalanced-tuple-unpacking
    popt, _ = curve_fit(decaying_fun, t_array, kinetic_energy,
                        p0=(max(kinetic_energy), gamma_0, omega_0, 0))
    delta_12 = (popt[2]-omega_0)/gamma_0
    gamma = popt[1]/gamma_0

    if print_values:
        print(r'\delta_12: ', delta_12)
        print(r'\gamma: ', gamma)
    if plot:
        plt.plot(t_array, kinetic_energy, label='KE')
        plt.plot(t_array, decaying_fun(t_array, *popt), '--', label='Fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Kinetic Energy (J)')
        plt.xlim(min(t_array), max(t_array))
        plt.legend()
        plt.show()
    return (delta_12, gamma)


def p_dipole_theory(
    r: float,
    d_12: float,
    omega_0: float,
    print_values: bool = False
) -> Tuple[float, float]:
    r"""Return \delta_12 and \gamma_12 of two identical coupled p-dipoles.

    Calculates results from Green's function theory. The results are scaled by
    gamma_0. Both dipoles must have identical initial dipole moments.

    Args:
        r: Distance between two charges in dipole.
        d_12: Distance between two dipoles.
        omega_0: Natural angular frequency of dipole.
        print_values: Prints \delta_12 and \gamma_12 if `True`.
            Defaults to `True`.

    Returns:
        \delta_12 and \gamma_12 scaled by \gamma_0.
    """
    k0 = omega_0/c
    lambda0 = 2*np.pi / k0
    eps = 1  # Free space
    G12ss = (k0**2*np.exp(1j*k0*d_12)/(4*np.pi*d_12)
             * (1+1j/(k0*d_12)*1-1/(k0*d_12)**2))
    G12pp = (G12ss + k0**2*np.exp(1j*k0*d_12)/(4*np.pi*d_12)
             * (-1-3j/(k0*d_12)*1+3/(k0*d_12)**2))
    G11 = 1j*omega_0**3/(6*np.pi*c**3)*np.sqrt(eps)
    gamma_0 = 2*r**2/(eps_0*hbar)*np.imag(G11)

    delta_12p = -r**2/(eps_0*hbar)*np.real(G12pp)/gamma_0
    gamma12p = 2*r**2/(eps_0*hbar)*np.imag(G12pp)/gamma_0
    if print_values:
        print('Separation (lambda_0): ', d_12/lambda0)
        print('delta_12 p: ', delta_12p)
        print('gamma12 p: ', gamma12p)
    return (delta_12p, gamma12p)


def s_dipole_theory(
    r: float,
    d_12: float,
    omega_0: float,
    print_values: bool = False
) -> Tuple[float, float]:
    r"""Return \delta_12 and \gamma_12 of two identical coupled s-dipoles.

    Calculates results from Green's function theory. The results are scaled by
    gamma_0. Both dipoles must have identical initial dipole moments.

    Args:
        r: Distance between two charges in dipole.
        d_12: Distance between two dipoles.
        omega_0: Natural angular frequency of dipole.
        print_values: Prints \delta_12 and \gamma_12 if `True`.
            Defaults to `True`.

    Returns:
        \delta_12 and \gamma_12 scaled by \gamma_0.
    """
    k0 = omega_0/c
    lambda0 = 2*np.pi / k0
    eps = 1  # Free space
    G12ss = (k0**2*np.exp(1j*k0*d_12)/(4*np.pi*d_12)
             * (1+1j/(k0*d_12)*1-1/(k0*d_12)**2))
    G11 = 1j*omega_0**3/(6*np.pi*c**3)*np.sqrt(eps)
    gamma_0 = 2*r**2/(eps_0*hbar)*np.imag(G11)

    delta_12s = -r**2/(eps_0*hbar)*np.real(G12ss)/gamma_0
    gamma12s = 2*r**2/(eps_0*hbar)*np.imag(G12ss)/gamma_0
    if print_values:
        print('Separation (lambda_0): ', d_12/lambda0)
        print('delta_12 s: ', delta_12s)
        print('gamma12 s: ', gamma12s)
    return (delta_12s, gamma12s)
