"""
Quickstart
==========

Welcome to the TorchOptics quickstart! This guide walks you through the main concepts of optical
simulations, including:

- Implementing optical fields using the :class:`~torchoptics.Field` class
- Propagating fields through free space
- Simulating lenses for focusing and imaging
- Simplifying simulations with the :class:`~torchoptics.System` class

Before starting, make sure TorchOptics is installed (:ref:`installation`).
"""

# %%
# Import PyCharge and other necessary libraries
# ----------------------------------------------
#
# We first import the necessary PyCharge components:

import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e

from pycharge import Charge, potentials_and_fields


# %% Define the position functions of a single charge
def position1(t):
    x = 0.0
    y = 0.0
    z = 0.0
    return x, y, z


charge1 = Charge(position1, e)
quantities_fn = potentials_and_fields([charge1])
# %% Define the observation grid in the x-y plane at z=0 and t=0
x = jnp.linspace(-1e-9, 1e-9, int(1e3))
y = jnp.linspace(-1e-9, 1e-9, int(1e3))
z = jnp.array([0])
t = jnp.array([0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")


# %% Calculate the scalar potential at grid points
quantities = quantities_fn(X, Y, Z, T)
electric_field_grid = quantities.electric
scalar_potential_grid = quantities.scalar

# %% Plot the potential along the observation grid
plt.figure()
plt.imshow(
    scalar_potential_grid.squeeze(),
    extent=(-1e-9, 1e-9, -1e-9, 1e-9),
    origin="lower",
    vmax=10,
    vmin=0,
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of a stationary charge")
plt.show()

fig, axes = plt.subplots(1, 3)
for i in range(3):
    axes[i].imshow(
        electric_field_grid[..., i].squeeze().T,
        # extent=(-1e-9, 1e-9, -1e-9, 1e-9),
        origin="lower",
        vmax=1e10,
        vmin=-1e10,
    )
    axes[i].set_axis_off()
    # axes[i].set_title(f"Electric Field Component E_{['x', 'y', 'z'][i]}")
    # axes[i].set_xlabel("X Position (m)")
    # axes[i].set_ylabel("Y Position (m)")

plt.suptitle("Electric Field of a Stationary Dipole")
plt.show()


# %%
# Stationary Dipole
# -----------------
# Now, let's simulate a stationary dipole consisting of two charges.
def position2(t):
    x = 5e-10
    y = 0.0
    z = 0.0
    return x, y, z


charge2 = Charge(position2, -e)
quantities_fn = potentials_and_fields([charge1, charge2])
# %% Calculate the scalar potential at grid points
# This calculates the scalar potential, vector potential, electric and magnetic fields.
quantities = quantities_fn(X, Y, Z, T)
scalar_potential_grid = quantities.scalar
electric_field_grid = quantities.electric

# %% Plot the potential along the observation grid
plt.imshow(
    scalar_potential_grid.squeeze().T,
    extent=(-1e-9, 1e-9, -1e-9, 1e-9),
    origin="lower",
    vmax=10,
    vmin=-10,
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of a Stationary Dipole")
plt.show()


# %%
# Oscillating Charge

amplitude = 1e-10
max_v = 0.8 * c
omega = max_v / amplitude


def position3(t):
    x = amplitude * jnp.sin(omega * t)
    y = 0.0
    z = 0.0
    return x, y, z


charge3 = Charge(position3, e)
quantities_fn = potentials_and_fields([charge3])
# %% Calculate the scalar potential at grid points
scalar_potential_grid = quantities_fn(X, Y, Z, T).scalar
# %% Plot the potential along the observation grid
plt.imshow(
    scalar_potential_grid.squeeze().T,
    extent=(-1e-9, 1e-9, -1e-9, 1e-9),
    origin="lower",
    vmax=10,
    vmin=0,
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of an Accelerating Charge")
plt.show()
