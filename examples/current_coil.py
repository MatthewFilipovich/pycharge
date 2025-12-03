"""
Visualizing Coil Fields in 2D
=============================

This example demonstrates how to visualize the complete electromagnetic response
from a current coil. We model the coil using discrete charges moving in a
circular trajectory and calculate all four electromagnetic quantities: scalar
potential, vector potential, electric field, and magnetic field on a 2D grid.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Create the charges for the current coil
# ----------------------------------------
#
# Coil parameters
num_charges = 20  # Number of discrete charges in the ring
R = 1e-2  # Coil radius (1 cm)
omega = 1e8  # Angular velocity (rad/s)


def get_circular_position(phi):
    """Returns a function for a circular trajectory with a given phase."""

    def position(t):
        x = R * jnp.cos(omega * t + phi)
        y = R * jnp.sin(omega * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(position_fn=get_circular_position(phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]


# %%
# Define the observation grid and calculate quantities
# -----------------------------------------------------

grid_res = 200
x = jnp.linspace(-1.5 * R, 1.5 * R, grid_res)
y = jnp.linspace(-1.5 * R, 1.5 * R, grid_res)
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Get the JIT-compiled function and calculate the quantities
fn = jax.jit(potentials_and_fields(charges))
output = fn(X, Y, Z, T)


def _sym_log_norm(data):
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0:
        return None
    return colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)


def _plot_scalar(field_data, title, extent):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(field_data.T, origin="lower", cmap="RdBu_r", norm=_sym_log_norm(field_data), extent=extent)
    fig.colorbar(im, ax=ax, label=f"{title} (a.u.)")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    plt.show()


def _plot_vector(field_data, title, extent):
    component_names = ("x", "y", "z")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, axis in enumerate(axes):
        component = field_data[..., idx]
        im = axis.imshow(
            component.T, origin="lower", cmap="RdBu_r", norm=_sym_log_norm(component), extent=extent
        )
        fig.colorbar(im, ax=axis, label="Field Strength")
        axis.set_title(f"{title} ({component_names[idx]}-component)")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.tight_layout()
    plt.show()


# %%
# Plot the results
# ----------------
#
# Loop through each electromagnetic quantity and create visualizations
extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

for field_name in ("scalar", "vector", "electric", "magnetic"):
    data = getattr(output, field_name).squeeze()
    title = field_name.capitalize()
    if data.ndim == 2:
        _plot_scalar(data, f"{title} Potential", extent)
    else:
        _plot_vector(data, f"{title} Field", extent)
