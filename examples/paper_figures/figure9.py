"""Module plots the dipole moment in frequency domain of 1 and 2 LOs."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

import pycharge as pc


def latex_plot(font_size=9, width=253):
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
    max_energy = max(max(energy1), max(energy2))  # pylint: disable=W3301
    return (energy1/max_energy, energy2/max_energy)


def fft(x, dt_val, padding):
    freq = np.fft.rfftfreq(int(len(x)*padding), dt_val) * 2*np.pi
    fft_sig = np.abs(np.fft.rfft(x, int(len(x)*padding)))
    return freq, fft_sig


latex_plot()

d1 = 2e-9
d12 = 80e-9
timesteps = 10000000
q = 50*e
origin = [(0, 0, 0), (0, d12, 0)]
omega_0 = 200e12*2*np.pi
dt = 1e-17

init_dipole = [(d1, 0, 0), (1e-15, 0, 0)]
charges = (
    pc.Dipole(omega_0, origin[0], init_dipole[0], q),
    pc.Dipole(omega_0, origin[1], init_dipole[1], q)
)
simulation = pc.Simulation(charges)
simulation.run(timesteps, dt, 'figure9.dat')
ang_freq, fft_signal = fft((charges[0].moment_disp[0]*q), dt, padding=1)
norm_fft = max(fft_signal)
fft_signal /= norm_fft

isolated_charge = pc.Dipole(omega_0, origin[0], init_dipole[0], q)
simulation_isolated = pc.Simulation(isolated_charge)
simulation_isolated.run(timesteps, dt, 'figure9.dat')
ang_freq, fft_signal_isolated = fft(
    (isolated_charge.moment_disp[0]*q), dt, padding=1)
fft_signal_isolated /= norm_fft

Delta12, Gamma12 = pc.s_dipole_theory(d1, d12, omega_0, True)
plt.semilogy((ang_freq-omega_0)/isolated_charge.gamma_0,
             fft_signal_isolated, label='1 LO')
plt.semilogy((ang_freq-omega_0)/charges[0].gamma_0, fft_signal, label='2 LOs')
plt.xlim(-50, 50)
plt.ylim(5.5e-3, 1.6)
plt.xlabel(r'$(\omega-\omega_0)/\gamma_0$')
plt.ylabel(r'$|\mathbf{d}|$ (arb. u.)')
plt.legend(handlelength=1.7, loc='upper center', bbox_to_anchor=(0.5, 1.21),
           ncol=2)
plt.savefig('figure9.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)
