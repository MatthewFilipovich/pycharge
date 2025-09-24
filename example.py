import numpy as np

import pycharge as pc

# Simulation variables
init_r = 1e-9  # Initial charge separation along y-axis
d_12 = 80e-9  # Distance between dipoles along x-axis
omega_0 = 100e12 * 2 * np.pi  # Natural angular frequency of the dipoles
timesteps = 40000  # Number of time steps in the simulation
dt = 1e-18  # Time step

origin_list = ((0, 0, 0), (d_12, 0, 0))  # Dipole origin vectors
init_r_list = ((0, init_r, 0), (0, init_r, 0))  # Initial charge displacements
sources = (
    pc.Dipole(omega_0, origin_list[0], init_r_list[0]),
    pc.Dipole(omega_0, origin_list[1], init_r_list[1]),
)
simulation = pc.Simulation(sources)
print("Running simulation...")

simulation.run(timesteps, dt, 's_dipoles.dat')
