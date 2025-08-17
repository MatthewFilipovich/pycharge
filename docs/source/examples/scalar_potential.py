"""
Scalar potential
=================
"""

from time import time

# %% Import necessary libraries
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e

from pycharge import Charge, scalar_potential

# %% Define observation grid in the x-y plane
x = jnp.linspace(-1e-9, 1e-9, int(1e3))
y = jnp.linspace(-1e-9, 1e-9, int(1e3))
z = jnp.array([0])
t = jnp.array([0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")


# %% Define the source position function
def position1(t):
    x = 1e-10 * jnp.sin(1e18 * t)
    y = 1e-10 * jnp.cos(1e18 * t)
    z = 0.0
    return x, y, z


def position2(t):
    x = 1e-10 * jnp.sin(1e18 * t)
    y = -1e-10 * jnp.cos(1e18 * t) - 5e-10
    z = 0.0
    return x, y, z


# Create a list of charges and JIT compile the scalar potential function
charges = [Charge(position1, e), Charge(position2, 0.2 * e)]
potentials_fn = jax.jit(scalar_potential(charges))

# Calculate scalar potential at grid points
start_t = time()
scalar_potential_grid = potentials_fn(X, Y, Z, T)
print(f"Time taken for batched evaluation: {time() - start_t:.4f} seconds")

# %% Plot the result
plt.imshow(
    scalar_potential_grid.squeeze(),
    extent=(-1e-9, 1e-9, -1e-9, 1e-9),
    origin="lower",
    vmax=10,
    cmap="viridis",
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of a Moving Point Charge")
plt.show()
