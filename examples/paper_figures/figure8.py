"""Module plots the normalized popoulations of two dipoles."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

import pycharge as pc


def latex_plot(font_size=9, width=259):
    fig_width_in = width / 72.27
    fig_height_in = fig_width_in * (5**.5-1)/2
    nice_fonts = {
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': font_size,
        'font.size': font_size,
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.figsize': (fig_width_in, fig_height_in)
    }
    plt.style.use(nice_fonts)


def scaled_total_energy(dipole1, dipole2):
    k = dipole1.omega_0**2 * dipole1.m_eff
    energy1 = (0.5*dipole1.m_eff * dipole1.moment_vel[0]**2
               + 0.5*k*(dipole1.moment_disp[0])**2)
    energy2 = (0.5*dipole2.m_eff * dipole2.moment_vel[0]**2
               + 0.5*k*(dipole2.moment_disp[0])**2)
    max_energy = max(max(energy1), max(energy2))
    return (energy1/max_energy, energy2/max_energy)


latex_plot()

d1 = 2e-9
d12 = 80e-9
timesteps = 10000000
q = 20*e
origin = [(0, 0, 0), (0, d12, 0)]
omega_0 = 200e12*2*np.pi
dt = 1e-17

init_dipole = [(d1, 0, 0), (1e-15, 0, 0)]
charges = (pc.Dipole(omega_0, origin[0], init_dipole[0], q),
           pc.Dipole(omega_0, origin[1], init_dipole[1], q))
simulation = pc.Simulation(charges)
simulation.run(timesteps, dt, 'figure8.dat')

Delta12, Gamma12 = pc.s_dipole_theory(d1, d12, omega_0, True)
t = dt*np.arange(0, timesteps) * charges[0].gamma_0
Ca = 1/4*(np.exp(t*(-1-Gamma12)) + np.exp(t*(-1+Gamma12))
          + 2*np.cos(2*Delta12*t)*np.exp(-t))
Cb = 1/4*(np.exp(t*(-1-Gamma12)) + np.exp(t*(-1+Gamma12))
          - 2*np.cos(2*Delta12*t)*np.exp(-t))

total_e1, total_e2 = scaled_total_energy(charges[0], charges[1])
plt.plot(t, total_e1, color='#1f77b4', label='Dipole $a$ Energy', zorder=8)
plt.plot(t, total_e2, color='#ff7f0e', label='Dipole $b$ Energy', zorder=7)
plt.plot(t, np.abs(Ca), '--', color='#d62728',
         label=r'Th. $\rho_{aa}$', zorder=8)
plt.plot(t, np.abs(Cb), '--', color='#2ca02c',
         label=r'Th. $\rho_{bb}$', zorder=7)
plt.legend(handlelength=1.7, loc='upper center', bbox_to_anchor=(0.5, 1.31),
           ncol=2)
plt.xlabel(r'$t\gamma_0$')
plt.ylabel('Scaled Populations')
plt.xlim(0, 0.7)
plt.ylim(0, 1)
plt.savefig('figure8.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)
