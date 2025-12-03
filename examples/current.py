"""
Current Loop vs. Biot-Savart Law
================================

This example demonstrates how to model a continuous current with discrete moving
charges. We arrange a number of charges in a circle, all moving with the same
angular velocity. This creates a steady-state current loop.

We then calculate the magnetic field along the central axis of the loop and
compare the result from PyCharge with the well-known analytical solution from
the Biot-Savart law for a current loop. This serves as a good validation of
the simulation approach for magnetostatics.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# Simulation parameters
num_charges = 50  # Number of discrete charges approximating the current
R = 1e-3  # Radius of the loop (1 mm)
omega = 1e6  # Angular velocity (rad/s)

# Define the observation axis (z-axis)
z_axis = jnp.linspace(-5 * R, 5 * R, 500)


# %%
# Create the charges for the current loop
# ----------------------------------------


def get_circular_position(omega, phi):
    """Returns a function for a circular trajectory with a given phase."""

    def position(t):
        x = R * jnp.cos(omega * t + phi)
        y = R * jnp.sin(omega * t + phi)
        z = 0
        return [x, y, z]

    return position


# Create a list of charges, spaced out by their phase `phi`
phases = jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
charges = [Charge(position_fn=get_circular_position(omega, phi), q=e) for phi in phases]


# %%
# Calculate the magnetic field with PyCharge
# -------------------------------------------

x = jnp.zeros_like(z_axis)
y = jnp.zeros_like(z_axis)
t = jnp.zeros_like(z_axis)

# Get the JIT-compiled function
potentials_and_fields_fn = jax.jit(potentials_and_fields(charges))
quantities = potentials_and_fields_fn(x, y, z_axis, t)

# Extract the z-component of the magnetic field
B_pycharge = quantities.magnetic[:, 2]


# %%
# Calculate the analytical solution
# ----------------------------------

current = num_charges * e * omega / (2 * jnp.pi)
B_biot_savart = mu_0 * current * R**2 / (2 * (z_axis**2 + R**2) ** (3 / 2))


# %%
# Compare the results
# -------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(z_axis, B_pycharge, label="PyCharge Result")
ax.plot(z_axis, B_biot_savart, "--", label="Biot-Savart Law (Analytical)")
ax.set_xlabel("Position along z-axis (m)")
ax.set_ylabel("Magnetic Field (T)")
ax.set_title("On-Axis Magnetic Field of a Current Loop")
ax.legend()
ax.grid(True)
fig.tight_layout()
plt.show()
