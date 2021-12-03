"""Module plots the E field and potential of a stationary dipole."""
# pragma pylint: disable=unexpected-keyword-arg, missing-function-docstring
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

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
grid_size = 1001
Emax = 1e-2
Vmax = 1e-0

log_scale = 1e-2
Emin = Emax*log_scale
Vmin = Vmax*log_scale

# Calculate and plot V
charges = (pc.StationaryCharge((10e-9, 0, 0), e),
           pc.StationaryCharge((-10e-9, 0, 0), -e))
simulation = pc.Simulation(charges)
coord = np.linspace(-lim, lim, grid_size)
x, y, z = np.meshgrid(coord, coord, 0, indexing='ij')
V = simulation.calculate_V(0, x, y, z)
V = V[:, :, 0]

fig, ax = plt.subplots()
ax.set_xlabel('$x$ (nm)')
ax.set_ylabel('$y$ (nm)')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
im = plt.imshow(V.T, origin='lower',
                extent=[-lim, lim, -lim, lim], vmax=7)
plt.xticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
plt.yticks([-lim, -lim/2, 0, lim/2, lim], [-50, -25, 0, 25, 50])
im.set_norm(mpl.colors.SymLogNorm(linthresh=Vmin, linscale=1,
                                  vmin=-Vmax, vmax=Vmax))

# Calculate and plot E arrows
grid_size_E = 21
lim_E = 52e-9
X, Y, Z = np.meshgrid(np.linspace(-lim_E, lim_E, grid_size_E),
                      np.linspace(-lim_E, lim_E, grid_size_E),
                      0, indexing='ij')
Ex, Ey, Ez = simulation.calculate_E(0, X, Y, Z)
Ex = Ex[:, :, 0]
Ey = Ey[:, :, 0]

u = Ex
v = Ey
r = np.power(np.add(np.power(u, 2), np.power(v, 2)), 0.5)
cb = fig.colorbar(im, fraction=0.046, pad=0.04,
                  ticks=(1, 1e-1, 1e-2, 0, -1e-2, -1e-1, -1e0))
mticks_vals = 0.32125*np.log10(np.arange(2, 10))
minorticks = np.concatenate((.3575+mticks_vals, .3575+0.32125+mticks_vals,
                             -.3575-mticks_vals, -.3575-0.32125-mticks_vals,
                             np.linspace(-.3575, .3575, 21)))
cb.ax.yaxis.set_ticks(minorticks, minor=True)

cb.ax.set_ylabel(r'$\Phi$ (V)', rotation=270, labelpad=12)
# Remove arrows at the point charge positions
u[12, 10] = 0
v[12, 10] = 0
u[8, 10] = 0
v[8, 10] = 0

Q = plt.quiver(X, Y, u/r, v/r, scale_units='xy', pivot='middle')
ax.scatter(-10e-9, 0, s=3, c='red', marker='o')
ax.scatter(10e-9, 0, s=3, c='red', marker='o')
plt.savefig('figure4.pdf', bbox_inches='tight', pad_inches=0.03, dpi=500)
