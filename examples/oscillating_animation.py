"""Module creates an animation of an oscillating point charge's fields."""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation

import pycharge as pc

lim = 50e-9
grid_size = 1000
x, y, z = np.meshgrid(np.linspace(-lim, lim, grid_size), 0,
                      np.linspace(-lim, lim, grid_size), indexing='ij')

charge = pc.OscillatingCharge(origin=(0, 0, 0), direction=(1, 0, 0),
                              amplitude=2e-9, omega=7.49e+16)
simulation = pc.Simulation(charge)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_position([0, 0, 1, 1])
# Initialie im plot
im = ax.imshow(np.zeros((grid_size, grid_size)), origin='lower',
               extent=[-lim, lim, -lim, lim], vmax=7)
ax.set_xticks([])
ax.set_yticks([])
im.set_norm(mpl.colors.LogNorm(vmin=1e5, vmax=1e8))

# Quiver plot
grid_size_quiver = 17
lim = 46e-9
x_quiver, y_quiver, z_quiver = np.meshgrid(
    np.linspace(-lim, lim, grid_size_quiver), 0,
    np.linspace(-lim, lim, grid_size_quiver), indexing='ij'
)
Q = ax.quiver(x_quiver, z_quiver,
              x_quiver[:, 0, :], z_quiver[:, 0, :], scale_units='xy')
pos = ax.scatter(charge.xpos(0), 0, s=5, c='red', marker='o')


def _update_animation(frame):
    text = f"\rProcessing frame {frame+1}/{n_frames}."
    sys.stdout.write(text)
    sys.stdout.flush()
    t = frame*dt
    E_total = simulation.calculate_E(t=t, x=x, y=y, z=z, pcharge_field='Total')
    u = E_total[0][:, 0, :]
    v = E_total[2][:, 0, :]
    im.set_data(np.sqrt(u**2+v**2).T)
    E_total = simulation.calculate_E(
        t=t, x=x_quiver, y=y_quiver, z=z_quiver, pcharge_field='Total')
    u = E_total[0][:, 0, :]
    v = E_total[2][:, 0, :]
    r = np.power(np.add(np.power(u, 2), np.power(v, 2)), 0.5)
    Q.set_UVC(u/r, v/r)
    pos.set_offsets((charge.xpos(t), 0))
    return im


def _init_animate():
    """Necessary for matplotlib animate."""
    pass  # pylint: disable=unnecessary-pass


n_frames = 36  # Number of frames in gif
dt = 2*np.pi/charge.omega/n_frames
ani = FuncAnimation(fig, _update_animation,
                    frames=n_frames, blit=False, init_func=_init_animate)
ani.save('oscillating_charge.gif',
         writer=animation.FFMpegWriter(fps=12), dpi=200)
