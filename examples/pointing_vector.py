"""
Poynting Vector Visualization
============================

This example demonstrates how to calculate and visualize the Poynting vector,
which represents the directional energy flux of an electromagnetic field.
The Poynting vector is defined as:

.. math::

   \mathbf{S} = \\frac{1}{\mu_0} \mathbf{E} \\times \mathbf{B}

We calculate the Poynting vector for an oscillating electric dipole by
computing the cross product of the electric and magnetic fields returned
by the :func:`potentials_and_fields` function.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Define the source: an oscillating dipole
# -----------------------------------------

amplitude = 5e-11  # 50 picometers
# Set frequency such that max velocity is 80% of the speed of light
omega = (0.8 * c) / amplitude


def pos_positive_charge(t):
    """Position of the positive charge."""
    x = amplitude * jnp.sin(omega * t)
    return x, 0.0, 0.0


def pos_negative_charge(t):
    """Position of the negative charge."""
    x = -amplitude * jnp.sin(omega * t)
    return x, 0.0, 0.0


# Create the two charges that form the dipole
charge1 = Charge(position_fn=pos_positive_charge, q=e)
charge2 = Charge(position_fn=pos_negative_charge, q=-e)


# %%
# Set up the calculation
# ----------------------

quantities_fn = potentials_and_fields([charge1, charge2])
jit_quantities_fn = jax.jit(quantities_fn)


# %%
# Define observation grid and calculate fields
# ---------------------------------------------
#
# Create a high-resolution 2D grid in the x-y plane
grid_size = 800
xy_max = 1e-9  # 1 nanometer
x_grid = jnp.linspace(-xy_max, xy_max, grid_size)
y_grid = jnp.linspace(-xy_max, xy_max, grid_size)
z_grid = jnp.array([0.0])
t_grid = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing="ij")

# Calculate all electromagnetic quantities on the grid
quantities = jit_quantities_fn(X, Y, Z, T)


# %%
# Calculate and plot the Poynting vector
# ---------------------------------------
#
# Compute S = (1/mu_0) * E × B
poynting_vector = jnp.cross(quantities.electric, quantities.magnetic, axis=-1) * (1 / mu_0)
poynting_magnitude = jnp.linalg.norm(poynting_vector, axis=-1)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    poynting_magnitude.squeeze().T,
    extent=(-xy_max, xy_max, -xy_max, xy_max),
    origin="lower",
    vmax=2e19,
    cmap="inferno",
)
fig.colorbar(im, ax=ax, label=r"$||\mathbf{S}||$ (W/m$^2$)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Poynting Vector Magnitude of an Oscillating Dipole")
fig.tight_layout()
plt.show()

# %%
