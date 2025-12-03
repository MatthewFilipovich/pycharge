"""Electric Field Visualization
============================

This example demonstrates how to visualize electromagnetic fields from charges
moving along complex trajectories. The charges follow epicycloid paths, and
we plot all four electromagnetic quantities (scalar potential, vector potential,
electric field, and magnetic field) on a 2D grid.
"""

# %%
# Import necessary libraries
# --------------------------
#
# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Define the position functions of the charges
# ---------------------------------------------
#
# Parameters for complex epicycloid trajectories
num_charges = 20
R1 = 1e-2  # Radius of the first circular component (1 cm)
R2 = 0.5e-2  # Radius of the second circular component (5 mm)
omega1 = 1e8  # Angular velocity of the first component (rad/s)
omega2 = 1e9  # Angular velocity of the second component (rad/s)


def get_circular_position(phi):
    def position(t):
        x = R1 * jnp.cos(omega1 * t + phi) + R2 * jnp.cos(omega2 * t + phi)
        y = R1 * jnp.sin(omega1 * t + phi) + R2 * jnp.sin(omega2 * t + phi)
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
#
# Create a 2D grid in the x-y plane at t=0
x = jnp.linspace(-1.5 * R1, 1.5 * R1, int(1e3))
y = jnp.linspace(-1.5 * R1, 1.5 * R1, int(1e3))
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Calculate electromagnetic quantities using JIT-compiled function
fn = jax.jit(potentials_and_fields(charges))
output = fn(X, Y, Z, T)


def _sym_log_norm(data):
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0:
        return None
    return colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)


extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))


# %%
# Plot the fields
# ---------------

for field_name in ("scalar", "electric", "vector", "magnetic"):
    data = getattr(output, field_name).squeeze()
    if data.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", norm=_sym_log_norm(data), extent=extent)
        fig.colorbar(im, ax=ax, label=f"{field_name.capitalize()} (a.u.)")
        ax.set_title(f"{field_name.capitalize()} Potential")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.tight_layout()
        plt.show()
    else:
        component_names = ("x", "y", "z")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, axis in enumerate(axes):
            component = data[..., idx]
            im = axis.imshow(
                component,
                origin="lower",
                cmap="RdBu_r",
                norm=_sym_log_norm(component),
                extent=extent,
            )
            fig.colorbar(im, ax=axis, label="Field Strength (a.u.)")
            axis.set_title(f"{field_name.capitalize()} ({component_names[idx]}-component)")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
        fig.tight_layout()
        plt.show()

# %%
