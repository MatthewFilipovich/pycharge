"""
Current coil 2
=================
"""


# %% Import necessary libraries

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.constants import e

from pycharge import Charge, quantities

# jax.config.update("jax_enable_x64", True)


# %% Define the position functions of the two charges
num_charges = 20
R1 = 1e-2
R2 = 0.5e-2
omega1 = 1e8
omega2 = 1e9


def get_circular_position(phi):
    def position(t):
        x = R1 * jnp.cos(omega1 * t + phi) + R2 * jnp.cos(omega2 * t + phi)
        y = R1 * jnp.sin(omega1 * t + phi) + R2 * jnp.sin(omega2 * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(get_circular_position(phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]

# %% Define the observation grid in the x-y plane at z=0 and t=0
x = jnp.linspace(-1.5 * R1, 1.5 * R1, int(1e3))
y = jnp.linspace(-1.5 * R1, 1.5 * R1, int(1e3))
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

fn = jax.jit(quantities(charges))
output = fn(X, Y, Z, T)

# %% Plot the potential along the observation grid
for key in ["scalar", "electric", "vector", "magnetic"]:
    if key == "scalar":
        fig, ax = plt.subplots()
        out = getattr(output, key).squeeze()
        if out.max() == 0:
            norm = None
        else:
            vmax = jnp.max(out).item()
            norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

        im = ax.imshow(out, origin="lower", cmap="RdBu_r", norm=norm)
        plt.colorbar(im, ax=ax)

    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            out = getattr(output, key).squeeze()[..., i]

            if out.max() == 0:
                norm = None
            else:
                vmax = jnp.max(out).item()
                norm = colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

            im = ax[i].imshow(out, origin="lower", cmap="RdBu_r", norm=norm)
            plt.colorbar(im, ax=ax[i])

    plt.suptitle(key)
    plt.tight_layout()
    plt.show()

# %%
