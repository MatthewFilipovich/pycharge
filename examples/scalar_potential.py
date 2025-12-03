"""
Scalar Potential from Multiple Charges
======================================

This example demonstrates how to calculate the scalar potential from a system
of multiple moving point charges. PyCharge applies the principle of
superposition, computing the contribution from each charge individually and
summing the results. Each charge can have its own unique trajectory and
charge value.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Define the trajectories for two different charges
# --------------------------------------------------
#
# First charge: circular trajectory centered at origin
def position1(t):
    """Circular trajectory for charge 1."""
    x = 1e-10 * jnp.sin(1e18 * t)
    y = 1e-10 * jnp.cos(1e18 * t)
    z = 0.0
    return x, y, z


def position2(t):
    """Offset circular trajectory for charge 2."""
    x = 1e-10 * jnp.sin(1e18 * t)
    y = -1e-10 * jnp.cos(1e18 * t) - 5e-10
    z = 0.0
    return x, y, z


# Create charges with different charge magnitudes
charges = [
    Charge(position_fn=position1, q=e),
    Charge(position_fn=position2, q=0.2 * e),
]


# %%
# Define the observation grid and calculate the potential
# --------------------------------------------------------
#
# Create JIT-compiled function for better performance
quantities_fn = jax.jit(potentials_and_fields(charges))

# Define a 2D observation grid in the x-y plane
grid_res = 200  # Grid resolution
xy_max = 1e-9  # Grid extends ±1 nanometer
x = jnp.linspace(-xy_max, xy_max, grid_res)
y = jnp.linspace(-xy_max, xy_max, grid_res)
z = jnp.array([0])
t = jnp.array([0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Calculate the scalar potential at grid points
scalar_potential_grid = quantities_fn(X, Y, Z, T).scalar


# %%
# Plot the resulting potential
# ----------------------------

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    scalar_potential_grid.squeeze().T,
    extent=(-xy_max, xy_max, -xy_max, xy_max),
    origin="lower",
    vmax=10,
    cmap="viridis",
)
fig.colorbar(im, ax=ax, label="Scalar Potential (V)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Scalar Potential of Two Moving Point Charges")
fig.tight_layout()
plt.show()
