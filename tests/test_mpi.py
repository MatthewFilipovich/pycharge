"""Unit tests for MPI functionality in pycharge package.

To run tests with MPI, use mpiexec on commandline:
>> mpiexec -n 2 python -m unittest test_mpi.py
"""
# pragma pylint: disable=missing-function-docstring, missing-class-docstring
import os
import unittest
from time import time

import numpy as np
from mpi4py import MPI  # pylint: disable=import-error

from pycharge import Dipole, Simulation, StationaryCharge


class Test(unittest.TestCase):

    def test_MPI_simulation(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, (1e-8, 0, 0), (-1e-9, 0, 0))
        simulation1 = Simulation((dipole1, dipole2))
        simulation1.run_mpi(100, 1e-18, save_E=False)
        dipole3 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        dipole4 = Dipole(100e12*2*np.pi, (1e-8, 0, 0), (-1e-9, 0, 0))
        simulation2 = Simulation((dipole3, dipole4))
        simulation2.run(100, 1e-18, save_E=False)
        np.testing.assert_equal(dipole1.moment_disp, dipole3.moment_disp)
        np.testing.assert_equal(dipole1.moment_vel, dipole3.moment_vel)
        np.testing.assert_equal(dipole2.moment_acc, dipole4.moment_acc)
        self.assertEqual(dipole1, dipole3)
        self.assertEqual(dipole2, dipole4)
        self.assertEqual(simulation1, simulation2)

    def test_load_save(self):
        try:
            dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
            charge1 = StationaryCharge((1e-7, 0, 0))
            simulation1 = Simulation((dipole1, charge1))
            simulation1.run_mpi(1000, 1e-18, 'test_mpi_save.dat')
            simulation2 = Simulation((dipole1, charge1))
            start_t = time()
            if MPI.COMM_WORLD.Get_rank() == 0:
                simulation2.run_mpi(
                    1000, 1e-18, 'test_mpi_save.dat')
                self.assertTrue((time()-start_t) < 0.05)
        finally:
            if MPI.COMM_WORLD.Get_rank() == 0:
                os.remove("test_mpi_save.dat")


if __name__ == '__main__':
    unittest.main()
