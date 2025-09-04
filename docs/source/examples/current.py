"""
Current
=================
"""

# %% Import necessary libraries
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.constants import e, mu_0

from pycharge import Charge, magnetic_field

# %%
num_charges = 2
R = 1e-3
omega = 1
z = jnp.linspace(0, 1e-3, 1000)


# %%
def get_circular_position(omega, phi):
    def position(t):
        x = R * jnp.cos(omega * t + phi)
        y = R * jnp.sin(omega * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(get_circular_position(omega, phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]
# %%
x = jnp.zeros_like(z)
y = jnp.zeros_like(z)
t = jnp.zeros_like(z)

magnetic_field_fn = jax.jit(magnetic_field(charges))
magnetic_field_grid = magnetic_field_fn(x, y, z, t)
# %% Calculate the magnetic field using Biot Savart
I = num_charges * e * omega / (2 * jnp.pi)
B_biot_savart = mu_0 * I * R**2 / (2 * (z**2 + R**2) ** (3 / 2))

# %%
plt.plot(z, magnetic_field_grid[:, 2])
plt.plot(z, B_biot_savart, "--")
