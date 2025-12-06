"""Simulate Coupled Dipoles
===========================

This example demonstrates the coherent oscillations between two nearby Lorentz
oscillators (dipoles). When placed in close proximity, the dipoles interact
through their electromagnetic fields, leading to synchronized motion. One dipole
is initialized in an excited state, while the other starts in its ground state.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e, m_e

from pycharge import dipole_source, simulate

jax.config.update("jax_enable_x64", True)


# %%
# Define dipoles and simulation timeline
# ---------------------------------------

initial_moment0 = [0.0, 0.0, 1e-9]  # Initial excited state along z-axis
initial_moment1 = [0.0, 0.0, -1e-20]  # Initial ground state along -z-axis
q = e * 20  # Charge magnitude for each dipole
omega_0 = 1e15 * 2 * jnp.pi  # 1 PHz natural frequency
m = m_e  # Electron mass
origin0 = [0.0, 0.0, 0.0]  # Origin of first dipole
origin1 = [0.0, 5e-9, 0.0]  # Origin of second dipole (5 nm apart)

# Create two identical dipoles separated along the y-axis
dipole0 = dipole_source(initial_moment0, omega_0, origin0, q, m)
dipole1 = dipole_source(initial_moment1, omega_0, origin1, q, m)

# Simulation time parameters
num_steps = 10_000
dt = 1e-17  # 10 attoseconds per step
ts = jnp.arange(num_steps) * dt


# %%
# Run the simulation
# ------------------

sim_fn = jax.jit(simulate([dipole0, dipole1], ts))
states = sim_fn()


# %%
# Plot the trajectory of one charge from each dipole
# ---------------------------------------------------

z_dipole0 = jnp.asarray(states[0][:, 0, 0, 2])
z_dipole1 = jnp.asarray(states[1][:, 0, 0, 2])

plt.figure(figsize=(9, 5))
plt.plot(ts, z_dipole0 * 1e9, label="Dipole 0 (Charge 0)")
plt.plot(ts, z_dipole1 * 1e9, label="Dipole 1 (Charge 0)")
plt.xlabel("Time (s)")
plt.ylabel("z-position (nm)")
plt.title("Coherent motion of coupled dipoles")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()
