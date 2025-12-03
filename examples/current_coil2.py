"""Visualizing Fields from Complex Trajectories
===============================================

This example demonstrates how PyCharge handles arbitrarily complex charge
trajectories. Each charge follows an epicycloid-like path created by
superposing two circular motions with different radii and frequencies.
Thanks to JAX's automatic differentiation, PyCharge can compute the fields
for any trajectory that can be expressed as a time-dependent function.
"""

# %%
# Import necessary libraries
# --------------------------

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# Define the complex trajectory
# ------------------------------
#
# Parameters for the epicycloid trajectory
num_charges = 20
R1 = 1e-2  # Radius of the first circular component (1 cm)
R2 = 0.5e-2  # Radius of the second circular component (5 mm)
omega1 = 1e8  # Angular velocity of the first component (rad/s)
omega2 = 1e9  # Angular velocity of the second component (rad/s)


def get_complex_position(phi):
    """Returns a function for a complex epicycloid-like trajectory."""

    def position(t):
        x = R1 * jnp.cos(omega1 * t + phi) + R2 * jnp.cos(omega2 * t + phi)
        y = R1 * jnp.sin(omega1 * t + phi) + R2 * jnp.sin(omega2 * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(position_fn=get_complex_position(phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]


# %%
# Define the observation grid and calculate quantities
# -----------------------------------------------------

grid_res = 200
x = jnp.linspace(-1.5 * R1, 1.5 * R1, grid_res)
y = jnp.linspace(-1.5 * R1, 1.5 * R1, grid_res)
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Get the JIT-compiled function and calculate the quantities
fn = jax.jit(potentials_and_fields(charges))
output = fn(X, Y, Z, T)


def _sym_log_norm(data):
    vmax = float(jnp.max(jnp.abs(data)))
    if vmax == 0:
        return None
    return colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)


def _plot_scalar(field_data, title, extent):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(field_data.T, origin="lower", cmap="RdBu_r", norm=_sym_log_norm(field_data), extent=extent)
    fig.colorbar(im, ax=ax, label=f"{title} (a.u.)")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    plt.show()


def _plot_vector(field_data, title, extent):
    component_names = ("x", "y", "z")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, axis in enumerate(axes):
        component = field_data[..., idx]
        im = axis.imshow(
            component.T, origin="lower", cmap="RdBu_r", norm=_sym_log_norm(component), extent=extent
        )
        fig.colorbar(im, ax=axis, label="Field Strength")
        axis.set_title(f"{title} ({component_names[idx]}-component)")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    fig.tight_layout()
    plt.show()


# %%
# Plot the results
# ----------------

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

for field_name in ("scalar", "vector", "electric", "magnetic"):
    data = getattr(output, field_name).squeeze()
    title = field_name.capitalize()
    if data.ndim == 2:
        _plot_scalar(data, f"{title} Potential", extent)
    else:
        _plot_vector(data, f"{title} Field", extent)
