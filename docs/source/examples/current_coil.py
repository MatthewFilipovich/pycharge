"""
Visualizing Coil Fields in 2D
=============================

This example demonstrates how to visualize the full electromagnetic response
(potentials and fields) from a current coil in a 2D plane.

We model a coil using a ring of discrete charges and then calculate the
scalar potential, vector potential, electric field, and magnetic field on a
2D grid that lies in the same plane as the coil. The results are plotted
as 2D images.
"""

# %%
# 1. Import necessary libraries
# -----------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# %%
# 2. Create the charges for the current coil
# ------------------------------------------
# We define a circular trajectory and create a list of `Charge` objects
# distributed evenly around the circle to model the coil.

num_charges = 20
R = 1e-2
omega = 1e8


def get_circular_position(phi):
    """Returns a function for a circular trajectory with a given phase."""

    def position(t):
        x = R * jnp.cos(omega * t + phi)
        y = R * jnp.sin(omega * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(get_circular_position(phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]

# %%
# 3. Define the observation grid and calculate quantities
# -------------------------------------------------------
# We define a 2D grid in the x-y plane at z=0 and t=0.

grid_res = 200
x = jnp.linspace(-1.5 * R, 1.5 * R, grid_res)
y = jnp.linspace(-1.5 * R, 1.5 * R, grid_res)
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Get the JIT-compiled function and calculate the quantities
fn = jax.jit(potentials_and_fields(charges))
output = fn(X, Y, Z, T)

# %%
# 4. Plot the results
# -------------------
# We loop through each calculated quantity and plot its components on the grid.
# A symmetric logarithmic scale is used for the color map to handle the large
# dynamic range of the field values, especially near the charges.

field_names = ["scalar", "vector", "electric", "magnetic"]
component_names = ["x", "y", "z"]

for key in field_names:
    # Squeeze to remove the singleton z and t dimensions
    field_data = getattr(output, key).squeeze()

    if key == "scalar":
        fig, ax = plt.subplots(figsize=(8, 7))
        vmax = jnp.max(jnp.abs(field_data)).item()
        norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

        im = ax.imshow(
            field_data.T, origin="lower", cmap="RdBu_r", norm=norm, extent=[x[0], x[-1], y[0], y[-1]]
        )
        fig.colorbar(im, ax=ax, label=f"{key.capitalize()} Potential (V)")
        ax.set_title(f"{key.capitalize()} Potential")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i in range(3):
            component_data = field_data[..., i]
            vmax = jnp.max(jnp.abs(component_data)).item()
            if vmax == 0:
                norm = None
            else:
                norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

            im = axes[i].imshow(
                component_data.T, origin="lower", cmap="RdBu_r", norm=norm, extent=[x[0], x[-1], y[0], y[-1]]
            )
            fig.colorbar(im, ax=axes[i], label="Field Strength")
            axes[i].set_title(f"{key.capitalize()} Field ({component_names[i]}-component)")
            axes[i].set_xlabel("x (m)")
            axes[i].set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()
