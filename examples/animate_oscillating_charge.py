"""Animate Oscillating Charge
=============================

This example demonstrates how to create an animation of the electric field
around a single oscillating charge. The visualization combines a heat map
showing the field magnitude with a quiver plot showing the field direction.
"""

# %%
# Import necessary libraries
# --------------------------
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


# %%
# Define the oscillating charge
# ------------------------------
#
# Amplitude and frequency of oscillation
amplitude = 2e-9  # 2 nanometers
omega = 7.5e16  # Angular frequency (rad/s)


def position(t):
    """Return the instantaneous position of the oscillating charge."""

    x = amplitude * jnp.cos(omega * t)
    return x, 0.0, 0.0


charge = Charge(position_fn=position, q=e)
quantities_fn = jax.jit(potentials_and_fields([charge]))


# %%
# Build the observation grids
# ----------------------------
#
# Create a high-resolution grid for the background field magnitude
field_limit = 50e-9  # 50 nanometers
field_grid_size = 300
x_vals = jnp.linspace(-field_limit, field_limit, field_grid_size)
y_vals = jnp.linspace(-field_limit, field_limit, field_grid_size)
z_vals = jnp.array([0.0])
X, Y, Z = jnp.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

# Create a coarser grid for the quiver plot arrows
quiver_limit = 46e-9  # Slightly smaller to avoid edge effects
quiver_grid_size = 21  # Fewer points for clearer arrow visualization
x_quiver = jnp.linspace(-quiver_limit, quiver_limit, quiver_grid_size)
y_quiver = jnp.linspace(-quiver_limit, quiver_limit, quiver_grid_size)
z_quiver = jnp.array([0.0])
XQ, YQ, ZQ = jnp.meshgrid(x_quiver, y_quiver, z_quiver, indexing="ij")


# %%
# Precompute helper quantities
# ----------------------------


def _normalized_vectors(vectors: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = jnp.where(norms == 0, 1.0, norms)
    return vectors / norms


initial_field = quantities_fn(X, Y, Z, jnp.zeros_like(X)).electric.squeeze()
initial_magnitude = np.asarray(jnp.linalg.norm(initial_field, axis=-1).T)

initial_quiver = quantities_fn(XQ, YQ, ZQ, jnp.zeros_like(XQ)).electric.squeeze()
initial_quiver_vecs = np.asarray(_normalized_vectors(initial_quiver))


# %%
# Animate the electric field
# --------------------------

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_position((0.0, 0.0, 1.0, 1.0))
ax.set_xticks([])
ax.set_yticks([])

im = ax.imshow(
    initial_magnitude,
    origin="lower",
    extent=(-field_limit, field_limit, -field_limit, field_limit),
    cmap="viridis",
)
im.set_norm(colors.LogNorm(vmin=1e5, vmax=1e8))

Q = ax.quiver(
    XQ[..., 0],
    YQ[..., 0],
    initial_quiver_vecs[..., 0],
    initial_quiver_vecs[..., 1],
    scale_units="xy",
)
pos = ax.scatter(float(position(0.0)[0]), float(position(0.0)[1]), s=10, c="red", marker="o")


def _update_animation(frame):
    text = f"\rProcessing frame {frame + 1}/{n_frames}."
    sys.stdout.write(text)
    sys.stdout.flush()

    t = frame * dt
    field_quantities = quantities_fn(X, Y, Z, jnp.full_like(X, t)).electric.squeeze()
    magnitude = np.asarray(jnp.linalg.norm(field_quantities, axis=-1).T)
    im.set_data(magnitude)

    quiver_quantities = quantities_fn(XQ, YQ, ZQ, jnp.full_like(XQ, t)).electric.squeeze()
    normalized_quiver = np.asarray(_normalized_vectors(quiver_quantities))
    u = normalized_quiver[..., 0]
    v = normalized_quiver[..., 1]
    Q.set_UVC(u, v)

    current_pos = position(t)
    pos.set_offsets((float(current_pos[0]), float(current_pos[1])))

    return [im, Q, pos]


def _init_animate() -> None:
    """Necessary for matplotlib animate."""
    pass  # pylint: disable=unnecessary-pass


n_frames = 32  # Number of frames in gif
dt = 2 * np.pi / omega / n_frames
FuncAnimation(fig, _update_animation, frames=n_frames, blit=False, init_func=_init_animate)
