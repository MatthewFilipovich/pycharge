"""
Quickstart
==========

Welcome to PyCharge! This guide provides a concise introduction to its core functionalities. We'll cover how to define a moving point charge and calculate its electromagnetic potentials and fields. Then, we'll show how to run a dynamic simulation of a Lorentz oscillator.

For detailed physical theory, please see the :doc:`/user_guide/index`.

.. _installation:

Before starting, make sure PyCharge is installed:

.. code-block:: bash

    pip install pycharge

---

Part 1: Calculating Potentials and Fields
-----------------------------------------

The primary function for calculating electromagnetic quantities is ``potentials_and_fields``. It takes a list of ``Charge`` objects and returns a new function that you can call with spacetime coordinates.

Let's see it in action.

"""

# %%
# 1. Import necessary libraries
# -----------------------------
# We import PyCharge's core components, JAX for numerical operations, and Matplotlib for plotting.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

# For better precision, you can enable 64-bit floating point numbers in JAX.
# jax.config.update("jax_enable_x64", True)


# %%
# 2. Define a Charge's Trajectory
# -------------------------------
# A charge's trajectory is defined by a simple Python function that takes time ``t``
# and returns the charge's ``[x, y, z]`` position. JAX's automatic differentiation
# will handle calculating the velocity and acceleration from this function.
#
# Here, we define a charge moving in a circle in the x-y plane.


def circular_position(t):
    """A function describing a circular trajectory."""
    radius = 1e-10
    omega = 1e16  # rad/s
    x = radius * jnp.cos(omega * t)
    y = radius * jnp.sin(omega * t)
    z = 0.0
    return x, y, z


# Create a Charge object with the defined trajectory and a charge value of +e.
moving_charge = Charge(position=circular_position, q=e)


# %%
# 3. Create the Calculation Function
# ----------------------------------
# We pass a list containing our charge to ``potentials_and_fields``.
# This returns a new, JAX-jittable function that is highly optimized for computing
# the potentials and fields.

quantities_fn = potentials_and_fields([moving_charge])

# For even better performance, we can explicitly JIT-compile it.
jit_quantities_fn = jax.jit(quantities_fn)


# %%
# 4. Define an Observation Grid and Calculate Quantities
# ------------------------------------------------------
# We'll define a 2D grid in the x-y plane to observe the fields at time t=0.
# The grid points are passed to our JIT-compiled function.

# Create a 2D grid (100x100 points)
grid_size = 100
x_grid = jnp.linspace(-5e-10, 5e-10, grid_size)
y_grid = jnp.linspace(-5e-10, 5e-10, grid_size)
z_grid = jnp.array([0.0])  # Observe in the z=0 plane
t_grid = jnp.array([0.0])  # Observe at t=0

# Use jnp.meshgrid to create the full 4D spacetime grid
X, Y, Z, T = jnp.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing="ij")

# Calculate all quantities (scalar/vector potentials, E/B fields) on the grid
quantities = jit_quantities_fn(X, Y, Z, T)


# %%
# 5. Visualize the Results
# ------------------------
# The output `quantities` is a `NamedTuple` containing JAX arrays for each
# physical quantity. Let's plot the scalar potential.

scalar_potential = quantities.scalar

plt.figure(figsize=(8, 6))
plt.imshow(
    scalar_potential.squeeze().T,
    extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
    origin="lower",
    cmap="viridis",
)
plt.colorbar(label="Scalar Potential (V)")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Scalar Potential of a Circularly Moving Charge at t=0")
plt.show()


# %%
#
# ---
#
# Part 2: Simulating Dynamic Sources
# ----------------------------------
#
# PyCharge can also simulate the dynamics of sources whose motion is governed by
# the electromagnetic fields they generate. This is done using the ``simulate``
# function, which takes a sequence of ``Source`` objects.
#
# Here, we simulate a dipole acting as a Lorentz oscillator.

from pycharge import dipole_source, simulate
from scipy.constants import m_e

# %%
# 1. Create a Dipole Source
# -------------------------
# We use the ``dipole_source`` factory to create a dipole. We define its
# initial charge separation, natural frequency, and other physical properties.

# A dipole with an initial 1nm separation along the z-axis.
dipole = dipole_source(
    d_0=[0.0, 0.0, 1e-9],
    q=e,
    omega_0=100e12 * 2 * jnp.pi,
    m=m_e,
)

# %%
# 2. Set Up and Run the Simulation
# --------------------------------
# We define the time steps for the simulation and then create the simulation
# function by passing a list of our sources to ``simulate``.

# Simulation time from 0 to 4e-14 seconds with 40,000 steps.
t_start = 0.0
t_num = 40_000
dt = 1e-18
ts = jnp.linspace(t_start, (t_num - 1) * dt, t_num)

# Create the simulation function and JIT-compile it
sim_fn = jax.jit(simulate([dipole]))

# Run the simulation
# The output is a tuple of states, one for each source.
source_states = sim_fn(ts)


# %%
# 3. Analyze the Simulation Results
# ---------------------------------
# The ``source_states`` contain the position and velocity of each charge in the
# source at every time step. Let's plot the z-position of the two charges
# in our dipole.

# The state for our first (and only) source
dipole_state = source_states[0]

# Extract the position array: shape is (num_timesteps, num_charges, 2, 3)
# The third dimension is for position (0) and velocity (1).
position_history = dipole_state[:, :, 0, :]

# Position history of the first charge in the dipole
charge1_pos = position_history[:, 0, :]
# Position history of the second charge in the dipole
charge2_pos = position_history[:, 1, :]

plt.figure(figsize=(10, 6))
plt.plot(ts, charge1_pos[:, 2], label="Charge 1 (negative)")
plt.plot(ts, charge2_pos[:, 2], label="Charge 2 (positive)")
plt.xlabel("Time (s)")
plt.ylabel("Z Position (m)")
plt.title("Oscillation of Charges in a Simulated Dipole")
plt.legend()
plt.grid(True)
plt.show()

# %%
# This quickstart has demonstrated the two main workflows in PyCharge:
#
# 1.  Calculating fields from charges with predefined trajectories.
# 2.  Simulating the dynamics of sources interacting with their own fields.
#
# Explore the :doc:`/examples/index` for more advanced use cases!
