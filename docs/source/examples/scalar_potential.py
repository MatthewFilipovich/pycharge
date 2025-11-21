"""
Scalar Potential from Multiple Charges
======================================

This example demonstrates how to calculate the scalar potential from a system
of multiple point charges. Each charge can have its own unique trajectory and
charge value.

PyCharge calculates the total potential by applying the principle of
superposition: it computes the potential from each charge individually and
then sums the results.
"""

# %%
# 1. Import necessary libraries
# -----------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)
# %%
# 2. Define the trajectories for two different charges
# ----------------------------------------------------
# The first charge moves in a circle. The second charge also moves in a
# circle but is offset on the y-axis.


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


# Create a list of charges with different charge values
charges = [Charge(position1, e), Charge(position2, 0.2 * e)]


# %%
# 3. Define the observation grid and calculate the potential
# ----------------------------------------------------------
# We create a JIT-compiled function for our system of charges and then
# call it on a 2D observation grid.

quantities_fn = jax.jit(potentials_and_fields(charges))

# Define the grid
grid_res = 200
xy_max = 1e-9
x = jnp.linspace(-xy_max, xy_max, grid_res)
y = jnp.linspace(-xy_max, xy_max, grid_res)
z = jnp.array([0])
t = jnp.array([0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Calculate the scalar potential at grid points
scalar_potential_grid = quantities_fn(X, Y, Z, T).scalar


# %%
# 4. Plot the resulting potential
# -------------------------------
# The plot shows the combined scalar potential from both moving charges.

plt.figure(figsize=(8, 6))
plt.imshow(
    scalar_potential_grid.squeeze().T,
    extent=(-xy_max, xy_max, -xy_max, xy_max),
    origin="lower",
    vmax=10,
    cmap="viridis",
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of Two Moving Point Charges")
plt.show()
