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
# 1. Import necessary libraries and define parameters
# ---------------------------------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e, mu_0

from pycharge import Charge, potentials_and_fields

# Simulation parameters
num_charges = 50  # Increase for a better approximation of a continuous current
R = 1e-3  # Radius of the loop
omega = 1e6  # Angular velocity of charges

# Define the observation axis (z-axis)
z_axis = jnp.linspace(-5 * R, 5 * R, 500)


# %%
# 2. Create the charges for the current loop
# ------------------------------------------
# We define a factory function that creates a circular trajectory. Then, we
# create a list of `Charge` objects distributed evenly around the circle.


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
charges = [Charge(get_circular_position(omega, phi), q=e) for phi in phases]


# %%
# 3. Calculate the Magnetic Field with PyCharge
# ---------------------------------------------
# We set up the observation points along the z-axis and call the JIT-compiled
# quantities function. We observe at t=0, as the current is in a steady state.

x = jnp.zeros_like(z_axis)
y = jnp.zeros_like(z_axis)
t = jnp.zeros_like(z_axis)

# Get the JIT-compiled function
potentials_and_fields_fn = jax.jit(potentials_and_fields(charges))
quantities = potentials_and_fields_fn(x, y, z_axis, t)

# Extract the z-component of the magnetic field
B_pycharge = quantities.magnetic[:, 2]


# %%
# 4. Calculate the Analytical Solution
# ------------------------------------
# The on-axis magnetic field for a current loop is given by the Biot-Savart law:
# B_z = (mu_0 * I * R^2) / (2 * (z^2 + R^2)^(3/2))

# The total current I is the charge per particle times the number of charges
# passing a point per second.
I = num_charges * e * omega / (2 * jnp.pi)
B_biot_savart = mu_0 * I * R**2 / (2 * (z_axis**2 + R**2) ** (3 / 2))


# %%
# 5. Compare the Results
# ----------------------
# We plot both the PyCharge result and the analytical solution. They should
# match very closely.

plt.figure(figsize=(8, 6))
plt.plot(z_axis, B_pycharge, label="PyCharge Result")
plt.plot(z_axis, B_biot_savart, "--", label="Biot-Savart Law (Analytical)")
plt.xlabel("Position along z-axis (m)")
plt.ylabel("Magnetic Field (T)")
plt.title("On-Axis Magnetic Field of a Current Loop")
plt.legend()
plt.grid(True)
plt.show()
