"""
Coupled Dipoles
================
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e, m_e

from pycharge import dipole_source, simulate

jax.config.update("jax_enable_x64", True)

dipole0 = dipole_source(d_0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 0.0, 0.0])
dipole1 = dipole_source(
    d_0=[0.0, 0.0, 1e-9], q=e, omega_0=100e12 * 2 * jnp.pi, m=m_e, origin=[0.0, 80e-9, 0.0]
)

t_start = 0.0
t_num = 5_000
dt = 1e-17
t_stop = (t_num - 1) * dt
ts = jnp.linspace(t_start, t_stop, t_num)

sim_fn = simulate([dipole0, dipole1], ts)
sim_fn = jax.jit(sim_fn)

state_list = sim_fn()

plt.plot(state_list[0][:, 0, 0, 2])
plt.plot(state_list[1][:, 0, 0, 2])
plt.show()
