"""Module simulates two coupled s-dipoles.

The two dipoles are separated along the x-axis by `d_12` and polarized along
the y-axis with an initial charge displacement of `d` for both dipoles. The
dipoles are intialized in-phase with each other.
"""
import numpy as np

import pycharge as pc

# Simulation variables
init_r = 1e-9  # Initial charge separation along y-axis
d_12 = 80e-9  # Distance between dipoles along x-axis
omega_0 = 100e12*2*np.pi  # Natural angular frequency of the dipoles
timesteps = 40000  # Number of time steps in the simulation
dt = 1e-18  # Time step

origin_list = ((0, 0, 0), (d_12, 0, 0))  # Dipole origin vectors
init_r_list = ((0, init_r, 0), (0, init_r, 0))  # Initial charge displacements
sources = (pc.Dipole(omega_0, origin_list[0], init_r_list[0]),
           pc.Dipole(omega_0, origin_list[1], init_r_list[1]))
simulation = pc.Simulation(sources)
# Simulation data of dipoles is saved to file 's_dipoles.dat'
simulation.run(timesteps, dt, 's_dipoles.dat')

# Calculate theoretical \delta and \gamma_12
theory_delta_12, theory_gamma_12 = pc.s_dipole_theory(init_r, d_12, omega_0)
# Calculate \delta and \gamma from simulation starting from 5000 time steps
delta_12, gamma = pc.calculate_dipole_properties(
    sources[0], first_index=10000, plot=True
)
print('Delta_12 (theory):', theory_delta_12)
print('Delta_12 (simulation):', delta_12)
print('Gamma_12 (theory):', theory_gamma_12)
print('Gamma (simulation):', gamma)
print()
