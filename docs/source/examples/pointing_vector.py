"""
Poynting Vector
===============

The Poynting vector represents the directional energy flux (the energy transfer
per unit area per unit time) of an electromagnetic field. It is defined as:

.. math::

   \mathbf{S} = \\frac{1}{\mu_0} \mathbf{E} \\times \mathbf{B}

In PyCharge, we can easily calculate the Poynting vector. The :func:`potentials_and_fields`
function returns a ``Quantities`` object that contains both the electric (``E``) and
magnetic (``B``) fields. We can then simply compute their cross product.

This example shows how to calculate and visualize the magnitude of the Poynting
vector for an oscillating electric dipole.
"""

# %%
# 1. Import necessary libraries
# -----------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# %%
# 2. Define the Source: An Oscillating Dipole
# -------------------------------------------
# We model a simple dipole by placing two opposite charges very close to each
# other and making them oscillate out of phase along the x-axis.

amplitude = 5e-11
# Set frequency such that max velocity is 90% of the speed of light
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
charge1 = Charge(position=pos_positive_charge, q=e)
charge2 = Charge(position=pos_negative_charge, q=-e)


# %%
# 3. Set up the Calculation
# -------------------------
# We create the quantities function and JIT-compile it for performance.

quantities_fn = potentials_and_fields([charge1, charge2])
jit_quantities_fn = jax.jit(quantities_fn)


# %%
# 4. Define Observation Grid and Calculate Fields
# -----------------------------------------------
# We'll observe the fields on a 2D grid in the x-y plane at t=0.

grid_size = 800
xy_max = 1e-9
x_grid = jnp.linspace(-xy_max, xy_max, grid_size)
y_grid = jnp.linspace(-xy_max, xy_max, grid_size)
z_grid = jnp.array([0.0])
t_grid = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing="ij")

# Calculate all electromagnetic quantities on the grid
quantities = jit_quantities_fn(X, Y, Z, T)


# %%
# 5. Calculate and Plot the Poynting Vector
# -----------------------------------------
# Now, we compute the cross product of E and B and plot its magnitude.
# Normalize the values to avoid floating point issues with very large/small numbers.

# S = (1/mu_0) * E x B
poynting_vector = jnp.cross(quantities.electric, quantities.magnetic, axis=-1) * (1 / mu_0)


# Calculate the magnitude
poynting_magnitude = jnp.linalg.norm(poynting_vector, axis=-1)
print(poynting_magnitude)

# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(
    poynting_magnitude.squeeze().T,
    extent=(-xy_max, xy_max, -xy_max, xy_max),
    origin="lower",
    vmax=2e19,  # Clip max value for better visualization
    cmap="inferno",
)

plt.colorbar(label=r"Poynting Vector Magnitude (normalized)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Poynting Vector of an Oscillating Dipole at t=0")
plt.show()

# %%
