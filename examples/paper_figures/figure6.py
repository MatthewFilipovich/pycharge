"""Calculates theoretical and simulated E fields from oscillating dipole."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, pi

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


def E_field_dipole(r_vector, t_val):
    r = np.linalg.norm(r_vector)
    r_mag = r_vector/r
    d = d0*np.exp(-1j*w*t_val)
    E = 1/(4*pi*epsilon_0) * (
        k**2 * np.cross(np.cross(r_mag, d), r_mag) * np.exp(1j*k*r)/r
        + (3*np.dot(r_mag, d)*r_mag - d) * (1/r**3 - 1j*k/r**2)*np.exp(1j*k*r))
    return E.real


# Simulation variables
w = 7e16  # Angular frequency of dipole
d0 = np.array((e*4e-9, 0, 0))  # Magnitude of dipole moment
num_points = 10000

wavelength = 2*pi*c/w
min_z = wavelength * 0.0001
max_z = wavelength * 10
t = 0
k = w/c

z_points = np.linspace(min_z, max_z, num_points)
Ex_theo = np.zeros(num_points)
for i, z_point in enumerate(z_points):
    Ex_theo[i] = E_field_dipole(np.array((0, 0, z_point)), t)[0]

charges = [pc.OscillatingCharge((0, 0, 0), (1, 0, 0), 2e-14, w, q=1e5*e),
           pc.OscillatingCharge((0, 0, 0), (-1, 0, 0), 2e-14, w, q=-1e5*e)]
simulation = pc.Simulation(charges)
x, y, z = np.meshgrid(0, 0, z_points, indexing='ij')
Ex, Ey, Ez = simulation.calculate_E(t, x, y, z)

# Plot figure
fig, axs = plt.subplots(2, sharex=True)
plt.subplots_adjust(hspace=0)

axs[0].plot(z_points/wavelength, Ex.flatten())
axs[0].set_yscale('symlog', linthresh=1e4, base=100)
axs[0].set_yticks((-1e16, -1e12, -1e8, -1e4, 1e4, 1e8))
axs[0].set_yticklabels(('$-10^{16}$', '$-10^{12}$', '$-10^{8}$',
                       '$-10^{4}$', '$10^{4}$', '$-10^{8}$'))
minor_t1 = np.linspace(-1e16, -1e12, 5, endpoint=True)
minor_t2 = np.linspace(-1e12, -1e8, 5, endpoint=True)
minor_t3 = np.linspace(-1e8, -1e4, 5, endpoint=True)
minor_t4 = np.linspace(-1e4, 1e4, 5, endpoint=True)
minor_t5 = np.linspace(1e4, 1e8, 5, endpoint=True)
axs[0].set_yticks(
    np.concatenate((minor_t1, minor_t2, minor_t3, minor_t4, minor_t5)),
    minor=True)
axs[0].set_ylim(-5e18, 1e8)

axs[1].plot(z_points/wavelength, 100*np.abs((Ex.flatten()-Ex_theo)/Ex_theo))
axs[1].set_yscale('log')
axs[1].set_xlim(-0.1, max(z_points/wavelength))
axs[1].set_yticks((1e-3, 1e-6, 1e-9, 1e-12))
minor_t1 = np.linspace(1e-6, 1e-3, 5, endpoint=True)
minor_t2 = np.linspace(1e-9, 1e-6, 5, endpoint=True)
minor_t3 = np.linspace(1e-12, 1e-9, 5, endpoint=True)
minor_t4 = np.linspace(1e-15, 1e-12, 5, endpoint=True)
axs[1].set_yticks(np.concatenate((minor_t1, minor_t2, minor_t3, minor_t4)),
                  minor=True)
axs[1].set_ylim(1e-14)

axs[1].set_xlabel(r'$z~(\lambda_0$)')
axs[0].set_ylabel('$E_x$ (N/C)')
axs[1].set_ylabel(r'Relative Error (\%)')

fig.align_ylabels(axs)
plt.savefig('figure6.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)
