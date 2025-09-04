"""
Current coil
=================
"""


# %% Import necessary libraries

# sphinx_gallery_start_ignore
# sphinx_gallery_multi_image = "single"
# sphinx_gallery_end_ignore

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import e

from pycharge import Charge, potentials_and_fields

# jax.config.update("jax_enable_x64", True)


# %% Define the position functions of the two charges
num_charges = 20
R = 1e-2
omega = 1e8


def get_circular_position(phi):
    def position(t):
        x = R * jnp.cos(omega * t + phi)
        y = R * jnp.sin(omega * t + phi)
        z = 0
        return [x, y, z]

    return position


charges = [
    Charge(get_circular_position(phi), q=e)
    for phi in jnp.linspace(0, 2 * jnp.pi, num_charges, endpoint=False)
]

# %% Define the observation grid in the x-y plane at z=0 and t=0
x = jnp.linspace(-1.5 * R, 1.5 * R, int(1e3))
y = jnp.linspace(-1.5 * R, 1.5 * R, int(1e3))
z = jnp.array([0.0])
t = jnp.array([0.0])

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

fn = jax.jit(potentials_and_fields(charges, scalar=True, vector=True, electric=True, magnetic=True))
output = fn(X, Y, Z, T)

# %% Plot the potential along the observation grid
for key in ["scalar", "electric", "vector", "magnetic"]:
    if key == "scalar":
        fig, ax = plt.subplots()
        out = output[key].squeeze()
        if out.max() == 0:
            norm = None
        else:
            vmax = jnp.max(out).item()
            norm = mpl.colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

        im = ax.imshow(out, origin="lower", cmap="RdBu_r", norm=norm)
        plt.colorbar(im, ax=ax)

    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            out = output[key].squeeze()[..., i]

            if out.max() == 0:
                norm = None
            else:
                vmax = jnp.max(out).item()
                norm = mpl.colors.SymLogNorm(linthresh=vmax * 1e-5, linscale=1, vmin=-vmax, vmax=vmax)

            im = ax[i].imshow(out, origin="lower", cmap="RdBu_r", norm=norm)
            plt.colorbar(im, ax=ax[i])

    plt.suptitle(key)
    plt.tight_layout()
    plt.show()

# %%
