"""Module plots modified radiative properties as function of separation."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import pycharge as pc


def latex_plot(font_size, width, height):
    fig_width_in = width / 72.27
    fig_height_in = height
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


latex_plot(font_size=9, width=258, height=3.328)

d = 1e-9  # Initial separation between charges in dipole (along x-axis)
omega_0 = 100e12*2*np.pi
timesteps = 100000
dt = 1e-18
first_KE = 50000  # First index of KE array used to calculate gamma and delta
d12_sweep = np.arange(50e-9, 10000e-9, 50e-9)  # Separations between dipoles

try:  # if previously calculated
    data = np.load('figure7.npz')
    deltas_s = data['deltas_s']
    gammas_s = data['gammas_s']
    deltas_s_theory = data['deltas_s_theory']
    gammas_s_theory = data['gammas_s_theory']
    deltas_p = data['deltas_p']
    gammas_p = data['gammas_p']
    deltas_p_theory = data['deltas_p_theory']
    gammas_p_theory = data['gammas_p_theory']
except FileNotFoundError:
    # Calculate s dipoles
    deltas_s = np.zeros(len(d12_sweep))
    gammas_s = np.zeros(len(d12_sweep))
    deltas_s_theory = np.zeros(len(d12_sweep))
    gammas_s_theory = np.zeros(len(d12_sweep))
    for i, d12 in enumerate(d12_sweep):
        print('Progress: ', i+1, '/', len(d12_sweep))
        origin_s = [(0, 0, 0), (0, d12, 0)]  # separated along y axis
        deltas_s_theory[i], gammas_s_theory[i] = pc.s_dipole_theory(
            d, d12, omega_0, False)
        sources = (pc.Dipole(omega_0, origin_s[0], (d, 0, 0)),
                   pc.Dipole(omega_0, origin_s[1], (d, 0, 0)))
        simulation = pc.Simulation(sources)
        simulation.run(timesteps, dt, progressbar=False)
        deltas_s[i], gammas_s[i] = pc.calculate_dipole_properties(
            sources[0], first_KE)

    # Calculate p dipoles
    deltas_p = np.zeros(len(d12_sweep))
    gammas_p = np.zeros(len(d12_sweep))
    deltas_p_theory = np.zeros(len(d12_sweep))
    gammas_p_theory = np.zeros(len(d12_sweep))
    for i, d12 in enumerate(d12_sweep):
        print('Progress: ', i+1, '/', len(d12_sweep))
        origin_p = [(0, 0, 0), (d12, 0, 0)]  # separated along x axis
        deltas_p_theory[i], gammas_p_theory[i] = pc.p_dipole_theory(
            d, d12, omega_0, False)
        sources = (pc.Dipole(omega_0, origin_p[0], (d, 0, 0)),
                   pc.Dipole(omega_0, origin_p[1], (d, 0, 0)))
        simulation = pc.Simulation(sources)
        simulation.run(timesteps, dt, progressbar=False)
        deltas_p[i], gammas_p[i] = pc.calculate_dipole_properties(
            sources[0], first_KE)
    np.savez(
        'figure7.npz', deltas_s=deltas_s, gammas_s=gammas_s,
        deltas_s_theory=deltas_s_theory, gammas_s_theory=gammas_s_theory+1,
        deltas_p=deltas_p, gammas_p=gammas_p, deltas_p_theory=deltas_p_theory,
        gammas_p_theory=gammas_p_theory+1,)
deltas_s_error = abs((deltas_s-deltas_s_theory)/deltas_s_theory*100)
gammas_s_error = abs((gammas_s-gammas_s_theory)/gammas_s_theory*100)
deltas_p_error = abs((deltas_p-deltas_p_theory)/deltas_p_theory*100)
gammas_p_error = abs((gammas_p-gammas_p_theory)/gammas_p_theory*100)

# Plot delta and gamma
lambda_0 = c/omega_0*2*np.pi
fig, axs = plt.subplots(2, sharex=True)
plt.subplots_adjust(hspace=0)

axs[0].plot(d12_sweep/lambda_0, deltas_p,
            color='#1f77b4', label='P Dipoles', zorder=1)
axs[0].plot(d12_sweep/lambda_0, deltas_s,
            color='#ff7f0e', label='S Dipoles', zorder=0)
axs[0].plot(d12_sweep/lambda_0, deltas_p_theory, '--',
            color='#d62728', label='Th. P Dipoles', zorder=1)
axs[0].plot(d12_sweep/lambda_0, deltas_s_theory,  '--',
            color='#2ca02c', label='Th. S Dipoles', zorder=0)
axs[0].set_yscale('symlog', linthresh=0.1)
axs[0].set_ylabel(r'$\delta_{12}$ ($\gamma_0$)')
axs[0].set_ylim(-3e3, 3e3)
axs[0].set_yticks((-1e3, -1e-1, -1e1, 1e-1, 1e1, 1e3))
minor_t0 = np.linspace(-1e-1, 1e-1, 5, endpoint=True)
minor_t1 = np.linspace(1e-1, 1e1, 5, endpoint=True)
minor_t2 = np.linspace(1e1, 1e3, 5, endpoint=True)
axs[0].set_yticks(
    np.concatenate((minor_t0, minor_t1, -minor_t1, minor_t2, -minor_t2)),
    minor=True)
axs[0].legend(handlelength=1.7, loc='upper center', bbox_to_anchor=(0.5, 1.4),
              ncol=2)

axs[1].plot(d12_sweep/lambda_0, gammas_p, color='#1f77b4', zorder=1)
axs[1].plot(d12_sweep/lambda_0, gammas_s, color='#ff7f0e', zorder=0)
axs[1].plot(d12_sweep/lambda_0, gammas_p_theory, '--',
            color='#d62728', zorder=1)
axs[1].plot(d12_sweep/lambda_0,  gammas_s_theory, '--',
            color='#2ca02c', zorder=0)
axs[1].set_xlabel(r'Separation ($\lambda_0$)')
axs[1].set_ylabel(r'$\gamma^+$ ($\gamma_0$)')
axs[1].set_xlim(0, max(d12_sweep/lambda_0))
axs[1].set_xticks(0.5*np.arange(7))
fig.align_ylabels(axs)

plt.savefig('figure7.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)

print('Average delta p relative error:', np.average(deltas_p_error))
print('Average delta s relative error:', np.average(deltas_s_error))
print('Average gamma p relative error:', np.average(gammas_p_error))
print('Average gamma s relative error:', np.average(gammas_s_error))
