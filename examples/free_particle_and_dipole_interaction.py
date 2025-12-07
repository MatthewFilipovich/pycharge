r"""Free Particle and Dipole Interaction
=======================================

Demonstrates a free charged particle responding to an oscillating dipole field.
The dipole (charges at :math:`z = \pm 0.5` nm) starts oscillating at :math:`t=0`.
The particle, located 1500 nm away, experiences:

1. **Initial motion**: Static electric field pushes particle downward (:math:`-z`).
2. **Propagation delay**: Oscillating radiation travels at speed of light :math:`c`.
   Delay time :math:`= d/c \approx 0.5` fs.
3. **Oscillations begin**: After delay, particle responds to time-varying
   electromagnetic wave with sinusoidal motion.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import c, e, m_e  # type: ignore

from pycharge import dipole_source, free_particle_source, simulate

jax.config.update("jax_enable_x64", True)


# %%
# Define sources and simulation timeline
# ---------------------------------------

# Oscillating dipole starting at t=0 with charges at z=±0.5 nm
initial_moment = [0.0, 0.0, 1e-9]
q_dipole = e * 50
omega_0 = 1e15 * 2 * jnp.pi  # 1 PHz
m_dipole = m_e
dipole_origin = [0.0, 0.0, 0.0]

dipole = dipole_source(initial_moment, omega_0, dipole_origin, q_dipole, m_dipole)

# Free particle at rest, 1500 nm away along x
particle_position = [15e-7, 0.0, 0]
q_particle = e
m_particle = m_e


def position_0_fn(t):
    """Initial position of the free particle."""
    return jnp.array(particle_position)


free_particle = free_particle_source(position_0_fn, q_particle, m_particle)

# Calculate light propagation delay
distance = jnp.linalg.norm(jnp.array(particle_position) - jnp.array(dipole_origin))
light_travel_time = distance / c

print(f"Distance from dipole to particle: {distance * 1e9:.1f} nm")
print(f"Light travel time: {light_travel_time * 1e15:.3f} fs")

# Simulation parameters
num_steps = 20_000
dt = 1e-18  # 1 attosecond
ts = jnp.arange(num_steps) * dt


# %%
# Run the simulation
# ------------------

sim_fn = jax.jit(simulate([dipole, free_particle], ts))
states = sim_fn()


# %%
# Extract and convert trajectories
# ---------------------------------

# Extract z-positions for all charges (in nm)
d1_z = states[0][:, 0, 0, 2] * 1e9  # Dipole charge 1 (-50e)
d2_z = states[0][:, 1, 0, 2] * 1e9  # Dipole charge 2 (+50e)
p_z = states[1][:, 0, 0, 2] * 1e9  # Free particle (+1e)


# %%
# Plot z-positions for all charges
# ---------------------------------
#
# **Dipole charges**: Oscillate at :math:`\omega_0 = 1` PHz along :math:`z`-axis
# around :math:`z = \pm 0.5` nm.
#
# **Free particle**: Initially drifts downward due to static field.
# After :math:`\sim 0.5` fs (light travel time), sinusoidal oscillations begin as the
# electromagnetic wave arrives. Red dashed line marks wave arrival.


def plot_z_position(ax, times, z_pos, color, title, vline_time=None):
    """Plot z-position with optional vertical line marker."""
    ax.plot(times, z_pos, color=color, linewidth=1.5)

    # Add vertical line if specified
    if vline_time is not None:
        ax.axvline(vline_time, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Light arrival")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_ylabel("z-position (nm)")
    ax.set_xlabel("Time (fs)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot z-position for each charge
ts_fs = ts * 1e15  # Time in femtoseconds

# Create titles with charge and position information
d1_title = f"Dipole Charge 1 (q={-q_dipole / e:.0f}e)\n(x={dipole_origin[0] * 1e9:.1f} nm, y={dipole_origin[1] * 1e9:.1f} nm)"
d2_title = f"Dipole Charge 2 (q={q_dipole / e:.0f}e)\n(x={dipole_origin[0] * 1e9:.1f} nm, y={dipole_origin[1] * 1e9:.1f} nm)"
p_title = f"Free Particle (q={q_particle / e:.0f}e)\n(x={particle_position[0] * 1e9:.1f} nm, y={particle_position[1] * 1e9:.1f} nm)"

plot_z_position(axes[0], ts_fs, d1_z, "C0", d1_title)
plot_z_position(axes[1], ts_fs, d2_z, "C1", d2_title)
plot_z_position(axes[2], ts_fs, p_z, "C2", p_title, vline_time=light_travel_time * 1e15)

plt.tight_layout()
plt.show()
