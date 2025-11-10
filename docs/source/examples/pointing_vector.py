"""
Poynting Vector
===============
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

# %% Define the observation grid in the x-y plane at z=0 and t=0
x = jnp.linspace(-1e-9, 1e-9, int(1e3))
y = jnp.linspace(-1e-9, 1e-9, int(1e3))
z = jnp.array([0])
t = jnp.array([0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")
# %% Define the position functions of a single charge
amplitude = 5e-11
max_v = 0.9 * c
omega = max_v / amplitude


def position1(t):
    x = amplitude * jnp.sin(omega * t)
    y = 0.0
    z = 0.0
    return x, y, z


def position2(t):
    x = -amplitude * jnp.sin(omega * t)
    y = 0.0
    z = 0.0
    return x, y, z


quantities_fn = potentials_and_fields([Charge(position1, e), Charge(position2, -e)])
# %% Calculate the scalar potential at grid points
quantities = quantities_fn(X, Y, Z, T)


poynting_vector_grid = jnp.cross(quantities.electric, quantities.magnetic, axis=-1)
poynting_vector_abs = jnp.linalg.norm(poynting_vector_grid, axis=-1)
# %% Plot the potential along the observation grid
plt.imshow(
    poynting_vector_abs.squeeze().T,
    extent=(-1e-9, 1e-9, -1e-9, 1e-9),
    origin="lower",
    vmax=1e13,
    # vmin=0,
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of an Accelerating Charge")
plt.show()

# %%
