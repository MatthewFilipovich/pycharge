"""
Visualizing Fields from Complex Trajectories
============================================

This example is similar to the `Current coil` example, but it demonstrates
how to visualize the fields from charges moving in more complex trajectories.

Here, the trajectory of each charge is a superposition of two circular motions
with different radii and frequencies, creating an epicycloid-like path.
PyCharge can handle any arbitrarily complex trajectory, as long as it can be
described by a time-dependent function, thanks to JAX's automatic
differentiation capabilities.
"""

# %%
# 1. Import necessary libraries
# -----------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)


# %%
# 2. Define the complex trajectory
# --------------------------------
# The position function is a sum of two ``cos`` and ``sin`` terms, creating a
# path that is more complex than a simple circle.

num_charges = 20
R1 = 1e-2
R2 = 0.5e-2
omega1 = 1e8
omega2 = 1e9


def get_complex_position(phi):
    """Returns a function for a complex epicycloid-like trajectory."""

    def position(t):
        x = R1 * jnp.cos(omega1 * t + phi) + R2 * jnp.cos(omega2 * t + phi)
        y = R1 * jnp.sin(omega1 * t + phi) + R2 * jnp.sin(omega2 * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(get_complex_position(phi), q=e) for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]

# %%
# 3. Define the observation grid and calculate quantities
# -------------------------------------------------------
grid_res = 200
x = jnp.linspace(-1.5 * R1, 1.5 * R1, grid_res)
y = jnp.linspace(-1.5 * R1, 1.5 * R1, grid_res)
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# Get the JIT-compiled function and calculate the quantities
fn = jax.jit(potentials_and_fields(charges))
output = fn(X, Y, Z, T)

# %%
# 4. Plot the results
# -------------------
# We loop through each calculated quantity and plot its components on the grid.

field_names = ["scalar", "vector", "electric", "magnetic"]
component_names = ["x", "y", "z"]

for key in field_names:
    # Squeeze to remove the singleton z and t dimensions
    field_data = getattr(output, key).squeeze()

    if key == "scalar":
        fig, ax = plt.subplots(figsize=(8, 7))
        vmax = jnp.max(jnp.abs(field_data)).item()
        norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

        im = ax.imshow(
            field_data.T, origin="lower", cmap="RdBu_r", norm=norm, extent=[x[0], x[-1], y[0], y[-1]]
        )
        fig.colorbar(im, ax=ax, label=f"{key.capitalize()} Potential (V)")
        ax.set_title(f"{key.capitalize()} Potential")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i in range(3):
            component_data = field_data[..., i]
            vmax = jnp.max(jnp.abs(component_data)).item()
            if vmax == 0:
                norm = None
            else:
                norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

            im = axes[i].imshow(
                component_data.T, origin="lower", cmap="RdBu_r", norm=norm, extent=[x[0], x[-1], y[0], y[-1]]
            )
            fig.colorbar(im, ax=axes[i], label="Field Strength")
            axes[i].set_title(f"{key.capitalize()} Field ({component_names[i]}-component)")
            axes[i].set_xlabel("x (m)")
            axes[i].set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()
