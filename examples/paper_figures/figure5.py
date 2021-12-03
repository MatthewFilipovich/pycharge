"""Module plots the Poynting vector from an oscillating dipole."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e, mu_0

import pycharge as pc


def latex_plot(font_size=9, width=330):
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


latex_plot()

lim = 50e-9
grid_size = 1000
Smax = 1e13
Smin = 1e10
omega = 7e16

# Calculate and plot E and B
charges = (pc.OscillatingCharge((0, 0, 0), (1, 0, 0), 2e-9,  omega, q=e),
           pc.OscillatingCharge((0, 0, 0), (-1, 0, 0), 2e-9,  omega, q=-e))
simulation = pc.Simulation(charges)
coord = np.linspace(-lim, lim, grid_size)
x, y, z = np.meshgrid(coord, coord, 0, indexing='ij')
Ex, Ey, _ = simulation.calculate_E(0, x, y, z, 'Acceleration')
_, _, Bz = simulation.calculate_B(0, x, y, z, 'Acceleration')

Ex = Ex[:, :, 0]
Ey = Ey[:, :, 0]
Bz = Bz[:, :, 0]

Sx = Ey*Bz
Sy = -Ex*Bz
S = (Sx**2+Sy**2)**0.5/mu_0

fig, ax = plt.subplots()
ax.set_xlabel('$x$ (nm)')
ax.set_ylabel('$y$ (nm)')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
im = plt.imshow(S.T, origin='lower',
                extent=[-lim, lim, -lim, lim], vmax=7)
plt.xticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
plt.yticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
im.set_norm(mpl.colors.LogNorm(vmin=Smin, vmax=Smax))

cb = fig.colorbar(im, fraction=0.046, pad=0.04)
cb.ax.set_ylabel(r'$|\mathbf{S}|$ (W/$\mathrm{m}^2$)',
                 rotation=270, labelpad=12)
ax.scatter(-2e-9, 0, s=3, c='red', marker='o')
ax.scatter(2e-9, 0, s=3, c='red', marker='o')
plt.savefig('figure5.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)
