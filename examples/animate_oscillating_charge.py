"""Animate Oscillating Charge
=============================

This example demonstrates how to create an animation of the electric field
around a single oscillating charge. The visualization combines a heat map
showing the field magnitude with a quiver plot showing the field direction.
"""

# %%
# Import necessary libraries
# --------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Define the oscillating charge
# ------------------------------
#
# Amplitude and frequency of oscillation
amplitude = 2e-9  # 2 nanometers
omega = 7.5e16  # Angular frequency (rad/s)


def position(t):
    """Return the instantaneous position of the oscillating charge."""
    return amplitude * jnp.cos(omega * t), 0.0, 0.0


charge = Charge(position_fn=position, q=e)
quantities_fn = jax.jit(potentials_and_fields([charge]))

# %%
# Build the observation grids
# ----------------------------

# High-resolution grid for background magnitude
limit = 50e-9  # 50 nanometers
n_points = 301  # Odd number ensures center point at (0, 0)
x = jnp.linspace(-limit, limit, n_points)
y = jnp.linspace(-limit, limit, n_points)

# Stride for quiver arrows along axes only
quiver_stride = 15

# %%
# Precompute all frames
# ---------------------

n_frames = 32
times = jnp.linspace(0, 2 * jnp.pi / omega, n_frames, endpoint=False)


def compute_fields(x_vals, y_vals, times):
    """Compute electric fields on a grid for all time frames.

    Returns arrays with shape (n_frames, n_y, n_x, 3) where the spatial
    dimensions are ordered as (y, x) to match matplotlib conventions.
    """
    t, y_grid, x_grid, z = jnp.meshgrid(times, y_vals, x_vals, jnp.array([0.0]), indexing="ij")
    return quantities_fn(x_grid, y_grid, z, t).electric.squeeze(axis=-2)


# Compute fields on both grids
E = compute_fields(x, y, times)  # (n_frames, n_y, n_x, 3)

# Extract magnitude and direction for quiver (subsampled)
E_magnitude = jnp.linalg.norm(E, axis=-1)  # (n_frames, n_y, n_x)
E_quiver = E[:, ::quiver_stride, ::quiver_stride, :]  # Subsample for arrows
E_quiver_norm = jnp.linalg.norm(E_quiver, axis=-1, keepdims=True)
E_quiver_dir = E_quiver / jnp.where(E_quiver_norm == 0, 1.0, E_quiver_norm)

# Charge x-position at each frame
charge_x = amplitude * jnp.cos(omega * times)


# %%
# Animate the electric field
# --------------------------

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_position((0.0, 0.0, 1.0, 1.0))
ax.axis("off")

im = ax.imshow(
    E_magnitude[0],
    origin="lower",
    extent=(-limit, limit, -limit, limit),
    cmap="viridis",
    norm=colors.LogNorm(vmin=1e5, vmax=1e8),
)

X, Y = jnp.meshgrid(x[::quiver_stride], y[::quiver_stride], indexing="xy")
quiver = ax.quiver(X, Y, E_quiver_dir[0, :, :, 0], E_quiver_dir[0, :, :, 1], scale_units="xy")
charge_marker = ax.scatter(charge_x[0], 0.0, s=10, c="red")


def _update_animation(frame):
    im.set_data(E_magnitude[frame])
    quiver.set_UVC(E_quiver_dir[frame, :, :, 0], E_quiver_dir[frame, :, :, 1])
    charge_marker.set_offsets([[charge_x[frame], 0.0]])
    return [im, quiver, charge_marker]


FuncAnimation(fig, _update_animation, frames=n_frames, blit=False)
