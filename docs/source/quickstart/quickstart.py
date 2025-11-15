"""
Quickstart
==========

Welcome to PyCharge! This guide provides a hands-on introduction to the core
features of this electromagnetics simulation library.

PyCharge has two primary workflows:

1.  **Point Charge Electromagnetics**: Compute relativistically-correct
    electromagnetic potentials and fields from point charges with predefined
    trajectories.
2.  **Self-Consistent N-Body Electrodynamics**: Simulate the dynamics of
    electromagnetic sources—such as dipoles—that interact through their
    self-generated fields.

This guide will walk you through both. For detailed physical theory, please see
the :doc:`/user_guide/index`.

.. _installation:

Before starting, make sure PyCharge is installed:

.. code-block:: bash

    pip install pycharge

Part 1: Point Charge Electrodynamics
------------------------------------

This workflow is for when you know the trajectory of a charge and want to
calculate the fields it produces. The primary function for this is
``potentials_and_fields``. It takes a list of ``Charge`` objects and returns a
new, highly-optimized function that you can call with spacetime coordinates.

Let's see it in action.

"""

# %%
# 1. Import necessary libraries
# -----------------------------
# We import PyCharge's core components, JAX for numerical operations, and
# Matplotlib for plotting.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e

from pycharge import Charge, potentials_and_fields

# For better precision, you can enable 64-bit floating point numbers in JAX.
jax.config.update("jax_enable_x64", True)


# %%
# 2. Define a Charge's Trajectory
# -------------------------------
# A charge's trajectory is defined by a simple Python function that takes time
# ``t`` and returns the charge's ``[x, y, z]`` position.
#
# A key feature of PyCharge is that it leverages JAX's automatic
# differentiation (``jax.jacobian``) to automatically calculate the velocity and
# acceleration from this position function. You only need to define the path!
#
# Here, we define a charge moving in a circle in the x-y plane.
circular_radius = 1e-10  # Circular radius of 0.1 nm
velocity = 0.9 * c  # Constant velocity of 90% the speed of light
omega = velocity / circular_radius  # Angular frequency


def circular_position(t):
    """A function describing a circular trajectory."""
    x = circular_radius * jnp.cos(omega * t)
    y = circular_radius * jnp.sin(omega * t)
    z = 0.0
    return x, y, z


# Create a Charge object with the defined trajectory and a charge value of +e.
moving_charge = Charge(position=circular_position, q=e)


# %%
# 3. Create the Calculation Function
# ----------------------------------
# We pass a list containing our charge to ``potentials_and_fields``. This
# returns a new function that is ready for JAX's just-in-time (JIT)
# compilation, making it extremely fast for repeated calculations.

quantities_fn = potentials_and_fields([moving_charge])

# For maximum performance, we explicitly JIT-compile it.
jit_quantities_fn = jax.jit(quantities_fn)


# %%
# 4. Define an Observation Grid and Calculate Quantities
# ------------------------------------------------------
# We'll define a 2D grid in the x-y plane to observe the fields at time t=0.
# The grid points are passed as JAX arrays to our JIT-compiled function.

# Create a 2D grid (1000x1000 points)
grid_size = 1000
xy_max = 5e-9
x_grid = jnp.linspace(-xy_max, xy_max, grid_size)
y_grid = jnp.linspace(-xy_max, xy_max, grid_size)
z_grid = jnp.array([0.0])  # Observe in the z=0 plane
t_grid = jnp.array([0.0])  # Observe at t=0

# Use jnp.meshgrid to create the full 4D spacetime grid
X, Y, Z, T = jnp.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing="ij")

# Calculate all quantities on the grid. This returns a `Quantities` object.
quantities = jit_quantities_fn(X, Y, Z, T)


# %%
# 5. Visualize the Results
# ------------------------
# The output `quantities` is a `NamedTuple` containing JAX arrays for the
# scalar potential, vector potential, electric field, and magnetic field.
#
# Let's plot the scalar potential and the magnitude of the electric field.

scalar_potential = quantities.scalar
electric_field = quantities.electric
electric_field_magnitude = jnp.linalg.norm(electric_field, axis=-1)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Scalar Potential
im1 = ax1.imshow(
    jnp.log10(scalar_potential.squeeze().T),
    extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
    origin="lower",
    cmap="viridis",
    vmax=1,
    vmin=-1,
)
fig.colorbar(im1, ax=ax1, label="Scalar Potential (V)")
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.set_title("Scalar Potential of a Circularly Moving Charge")

# Plot Electric Field Magnitude
im2 = ax2.imshow(
    (electric_field_magnitude.squeeze().T),
    extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()),
    origin="lower",
    cmap="inferno",
    vmax=1e10,
    vmin=0,
)
fig.colorbar(im2, ax=ax2, label="Electric Field Magnitude (V/m)")
ax2.set_xlabel("X Position (m)")
ax2.set_ylabel("Y Position (m)")
ax2.set_title("Electric Field of a Circularly Moving Charge")

plt.tight_layout()
plt.show()


# %%
#
# ---
#
# Part 2: Self-Consistent N-Body Electrodynamics
# ----------------------------------------------
#
# PyCharge can also simulate the dynamics of sources whose motion is governed
# by the electromagnetic fields they and other sources generate. This is done
# using the ``simulate`` function, which takes a sequence of ``Source`` objects
# and solves the underlying ordinary differential equations (ODEs).
#
# Here, we simulate a dipole modeled as a **Lorentz oscillator**, a classical
# analog for a two-level quantum system.

from scipy.constants import m_e

from pycharge import dipole_source, simulate

# %%
# 1. Create a Dipole Source
# -------------------------
# We use the ``dipole_source`` factory to create a dipole. This object bundles
# the initial state of the charges with the ODE that governs its motion. We
# define its initial charge separation, natural frequency, and other physical
# properties.

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
sim_fn = jax.jit(simulate([dipole], ts))

# Run the simulation. The output is a tuple of states, one for each source.
# Each state contains the full position and velocity history of its charges.
source_states = sim_fn()


# %%
# 3. Analyze the Simulation Results
# ---------------------------------
# The ``source_states`` contain the position and velocity of each charge in the
# source at every time step. Let's plot the z-position of the two charges
# in our dipole.
#
# The plot shows a damped oscillation. The dipole loses energy over time because,
# as an accelerating source, it radiates electromagnetic waves. This effect,
# known as **radiation damping**, is automatically captured by the simulation.

# The state for our first (and only) source
dipole_state = source_states[0]

# Extract the position history array: shape is (num_timesteps, num_charges, 2, 3)
# The third dimension is for position (0) and velocity (1).
position_history = dipole_state[:, :, 0, :]

# Position history of the first charge in the dipole (negative)
charge1_pos = position_history[:, 0, :]
# Position history of the second charge in the dipole (positive)
charge2_pos = position_history[:, 1, :]

plt.figure(figsize=(10, 6))
plt.plot(ts, charge1_pos[:, 2], label="Charge 1 (negative)")
plt.plot(ts, charge2_pos[:, 2], label="Charge 2 (positive)")
plt.xlabel("Time (s)")
plt.ylabel("Z Position (m)")
plt.title("Damped Oscillation of Charges in a Simulated Lorentz Dipole")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Next Steps
# ==========
#
# This quickstart has demonstrated the two main workflows in PyCharge:
#
# 1.  Calculating fields from charges with predefined trajectories.
# 2.  Simulating the dynamics of sources interacting with their own fields.
#
# To dive deeper, explore the :doc:`/user_guide/index` for more detailed
# explanations of the physics and the :doc:`/examples/index` for more
# advanced use cases!
# %%
