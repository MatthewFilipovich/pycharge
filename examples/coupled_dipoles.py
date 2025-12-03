"""Coupled Dipoles
=================

This example demonstrates the coherent oscillations between two nearby Lorentz
oscillators (dipoles). When placed in close proximity, the dipoles interact
through their electromagnetic fields, leading to synchronized motion.
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
#
# Physical parameters for the dipole system
separation = 80e-9  # 80 nanometers between dipoles
frequency = 100e12 * 2 * jnp.pi  # 100 THz natural frequency
initial_offset = [0.0, 0.0, 1e-9]  # Initial displacement along z-axis

# Create two identical dipoles separated along the y-axis
dipole0 = dipole_source(d_0=initial_offset, q=e, omega_0=frequency, m=m_e, origin=[0.0, 0.0, 0.0])
dipole1 = dipole_source(d_0=initial_offset, q=e, omega_0=frequency, m=m_e, origin=[0.0, separation, 0.0])

# Simulation time parameters
num_steps = 5_000
dt = 1e-17  # 10 attoseconds per step
ts = jnp.linspace(0.0, (num_steps - 1) * dt, num_steps)


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
