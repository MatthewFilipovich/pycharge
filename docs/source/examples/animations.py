"""
Animation
=========
"""
# type: ignore
# %%

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

amplitude = 2e-9
omega = 7.49e16


def position(t):
    """Position of the positive charge."""
    x = amplitude * jnp.sin(omega * t)
    return x, 0.0, 0.0


# Create the two charges that form the dipole
charge = Charge(position=position, q=e)
quantities_fn = potentials_and_fields([charge])
jit_quantities_fn = jax.jit(quantities_fn)

# # Observation grid

lim = 50e-9
grid_size = 1000
x, y, z = np.meshgrid(np.linspace(-lim, lim, grid_size), np.linspace(-lim, lim, grid_size), 0, indexing="ij")


fig, ax = plt.subplots(figsize=(5, 5))
ax.set_position((0.0, 0.0, 1.0, 1.0))
# Initialie im plot
im = ax.imshow(
    np.zeros((grid_size, grid_size)), origin="lower", extent=(-lim, lim, -lim, lim), vmax=7, cmap="viridis"
)
ax.set_xticks([])
ax.set_yticks([])
im.set_norm(colors.LogNorm(vmin=1e5, vmax=1e8))

# Quiver plot
grid_size_quiver = 17
lim = 46e-9
x_quiver, y_quiver, z_quiver = np.meshgrid(
    np.linspace(-lim, lim, grid_size_quiver), np.linspace(-lim, lim, grid_size_quiver), 0, indexing="ij"
)
Q = ax.quiver(x_quiver, y_quiver, x_quiver[:, :, 0], y_quiver[:, :, 0], scale_units="xy")
pos = ax.scatter(position(0)[0], position(0)[1], s=5, c="red", marker="o")


def _update_animation(frame):
    text = f"\rProcessing frame {frame + 1}/{n_frames}."
    sys.stdout.write(text)
    sys.stdout.flush()
    t = frame * dt
    E_total = jit_quantities_fn(x, y, z, jnp.full_like(x, t)).electric.squeeze()
    E_total_magnitude = jnp.linalg.norm(E_total, axis=-1)
    im.set_data(E_total_magnitude.T)

    E_total = jit_quantities_fn(x_quiver, y_quiver, z_quiver, jnp.full_like(x_quiver, t)).electric
    u = E_total[..., 0]
    v = E_total[..., 1]
    r = jnp.linalg.norm(E_total, axis=-1)
    Q.set_UVC(u / r, v / r)
    pos.set_offsets((position(0)[0], position(0)[1]))  # type: ignore
    return im


def _init_animate() -> None:
    """Necessary for matplotlib animate."""
    pass  # pylint: disable=unnecessary-pass


n_frames = 36  # Number of frames in gif
dt = 2 * np.pi / omega / n_frames
FuncAnimation(fig, _update_animation, frames=n_frames, blit=False, init_func=_init_animate)  # type: ignore
