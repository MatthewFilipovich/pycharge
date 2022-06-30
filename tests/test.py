"""Unit tests for pycharge package."""
# pragma pylint: disable=missing-function-docstring, missing-class-docstring
import os
import unittest
from time import time

import numpy as np
from scipy.constants import m_e

import pycharge as pc
from pycharge import (Dipole, LinearAcceleratingCharge,
                      LinearDeceleratingCharge, LinearVelocityCharge,
                      OrbittingCharge, OscillatingCharge, Simulation,
                      StationaryCharge)


class Test(unittest.TestCase):

    def test_all_charges(self):
        dipole = Dipole(100e12*2*np.pi, (1e-6, 1e-6, 1e-6), (1e-9, 0, 0))
        stationary = StationaryCharge((1e-9, 1e-9, 1e-9))
        oscillating = OscillatingCharge((1e-9, 2e-9, 5e-9), (1, 0, 0),
                                        1e-9, 2*np.pi*1e12, True, 1e-16)
        orbitting = OrbittingCharge(1e-9, 5*2*np.pi, True, 2e-15)
        linear_accelerating = LinearAcceleratingCharge(104324, 1e-16)
        linear_accelerating = LinearAcceleratingCharge(104324)
        linear_decelerating = LinearDeceleratingCharge(1324, 901.9, 1e-16)
        linear_decelerating = LinearDeceleratingCharge(1324, 901.9,)
        linear_velocity = LinearVelocityCharge(3828.2, 6e-9)
        simulation1 = Simulation((dipole, stationary, oscillating, orbitting,
                                  linear_accelerating, linear_decelerating,
                                  linear_velocity))
        simulation1.run(100, 1e-18)
        simulation2 = Simulation((dipole, stationary, oscillating, orbitting,
                                  linear_accelerating, linear_decelerating,
                                  linear_velocity))
        simulation2.run(100, 1e-18)
        self.assertEqual(simulation1, simulation2)

    def test_dipoles_equal(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        self.assertEqual(dipole1, dipole2)

    def test_dipoles_not_equal(self):
        dipole1 = Dipole(50e12*2*np.pi, (1e-9, 0, 0), (2e-9, 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        self.assertNotEqual(dipole1, dipole2)

    def test_dipole_raise_error(self):
        self.assertRaises(
            ValueError, Dipole, 100e12*2*np.pi, (0, 0, 0), (0, 0, 0)
        )

    def test_simulation_dipole(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation1 = Simulation(dipole1)
        simulation1.run(100, 1e-18)
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation2 = Simulation(dipole2)
        simulation2.run(100, 1e-18)
        np.testing.assert_equal(dipole1.moment_disp, dipole2.moment_disp)
        np.testing.assert_equal(dipole1.moment_vel, dipole2.moment_vel)
        np.testing.assert_equal(dipole1.moment_acc, dipole2.moment_acc)

    def test_multiple_dipoles(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, (1e-8, 0, 0), (-1e-9, 0, 0))
        dipole3 = Dipole(100e12*2*np.pi, (2e-8, 0, 0), (-1e-9, 0, 0))
        simulation1 = Simulation((dipole1, dipole2, dipole3))
        self.assertIsNone(simulation1.run(500, 1e-18))  # Check runs

    def test_charge_equal(self):
        charge1 = StationaryCharge((1e-7, 0, 0))
        charge2 = StationaryCharge((1e-7, 0, 0))
        self.assertEqual(charge1, charge2)

    def test_equal_simulation(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        charge1 = StationaryCharge((1e-7, 0, 0))
        simulation1 = Simulation((dipole1, charge1))
        simulation1.run(100, 1e-18)
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        charge2 = StationaryCharge((1e-7, 0, 0))
        simulation2 = Simulation((dipole2, charge2))
        simulation2.run(100, 1e-18)
        self.assertEqual(simulation1, simulation2)

    def test_simulation_dipole_charge(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        charge1 = StationaryCharge((1e-7, 0, 0))
        simulation1 = Simulation((dipole1, charge1))
        simulation1.run(100, 1e-18)
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        charge2 = StationaryCharge((1e-7, 0, 0))
        simulation2 = Simulation((dipole2, charge2))
        simulation2.run(100, 1e-18)
        np.testing.assert_equal(dipole1.moment_disp, dipole2.moment_disp)
        np.testing.assert_equal(dipole1.moment_vel, dipole2.moment_vel)
        np.testing.assert_equal(dipole1.moment_acc, dipole2.moment_acc)

    def test_moving_origin(self):
        def fun_origin1(t):
            return np.array((.1e-9*np.cos(1e12*t), 0, 0))
        dipole1 = Dipole(100e12*2*np.pi, fun_origin1, (1e-9, 0, 0))
        simulation1 = Simulation(dipole1)
        simulation1.run(100, 1e-18)

        def fun_origin2(t):
            return np.array((.1e-9*np.cos(1e12*t), 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, fun_origin2, (1e-9, 0, 0))
        simulation2 = Simulation(dipole2)
        simulation2.run(100, 1e-18)
        self.assertEqual(simulation1, simulation2)

        def fun_origin3(t):
            return np.array((.05e-9*np.cos(1e12*t), 0, 0))
        dipole3 = Dipole(100e12*2*np.pi, fun_origin3, (1e-9, 0, 0))
        simulation3 = Simulation(dipole3)
        simulation3.run(100, 1e-18)
        self.assertNotEqual(simulation2, simulation3)

    def test_load_save(self):
        try:
            dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
            charge1 = StationaryCharge((1e-7, 0, 0))
            simulation1 = Simulation((dipole1, charge1))
            simulation1.run(500, 1e-18, 'test_save.dat')
            simulation2 = Simulation((dipole1, charge1))
            start_t = time()
            simulation2.run(500, 1e-18, 'test_save.dat')
            self.assertTrue((time()-start_t) < 0.05)
        finally:
            os.remove('test_save.dat')

    def test_calculate_B_and_potential(self):
        charge1 = StationaryCharge((1e-7, 0, 0))
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation1 = Simulation((dipole1, charge1))
        simulation1.run(100, 1e-18)
        x, y, z = np.meshgrid(1e-8, 1e-8, 1e-8, indexing='ij')
        B1 = simulation1.calculate_B(12.1e-18, x, y, z)
        V1 = simulation1.calculate_V(12.1e-18, x, y, z)
        A1 = simulation1.calculate_A(12.1e-18, x, y, z)
        charge2 = StationaryCharge((1e-7, 0, 0))
        dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation2 = Simulation((dipole2, charge2))
        simulation2.run(100, 1e-18)
        B2 = simulation2.calculate_B(12.1e-18, x, y, z)
        V2 = simulation2.calculate_V(12.1e-18, x, y, z)
        A2 = simulation2.calculate_A(12.1e-18, x, y, z)
        np.testing.assert_equal(B1, B2)
        np.testing.assert_equal(V1, V2)
        np.testing.assert_equal(A1, A2)
        charge3 = StationaryCharge((1e-7, 0, 0))
        dipole3 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation3 = Simulation((dipole3, charge3))
        simulation3.run(100, 1e-18)
        B3 = simulation3.calculate_B(12.1e-18, x, y, z)
        V3 = simulation3.calculate_V(12.1e-18, x, y, z)
        np.any(np.not_equal(B1, B3))
        np.any(np.not_equal(V1, V3))
        np.any(np.not_equal(V1, V3))

    def test_combine_files(self):
        try:
            dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
            simulation1 = Simulation(dipole1)
            simulation1.run(501, 1e-18, 'test1.dat')
            dipole2 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
            simulation2 = Simulation(dipole2)
            simulation2.run(500, 1e-18, 'test2.dat')
            pc.combine_files(('test1.dat', 'test2.dat'), 'test3.dat')
            self.assertEqual(pc.get_file_length('test3.dat'), 2)

            start_t1 = time()
            simulation1.run(500, 1e-18, 'test3.dat')
            self.assertTrue((time()-start_t1) < 0.05)

            start_t2 = time()
            simulation2.run(600, 1e-18, 'test3.dat')  # Not saved
            self.assertFalse((time()-start_t2) < 0.05)
            self.assertTrue(simulation1, simulation2)
        finally:
            os.remove('test1.dat')
            os.remove('test2.dat')
            os.remove('test3.dat')

    def test_dipole_properties_vacuum(self):
        dipole1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0))
        simulation1 = Simulation(dipole1)
        simulation1.run(5000, 1e-18, save_E=True)
        # Calculate theoretical \delta_12 and \gamma_12
        delta_12, gamma = pc.calculate_dipole_properties(dipole1, 0)
        np.testing.assert_almost_equal(gamma, 1, 5)
        np.testing.assert_almost_equal(delta_12, 0, 5)

    def test_s_dipole_properties(self):
        origin_list = ((0, 0, 0), (80e-9, 0, 0))
        initial_d_list = ((0, 1e-9, 0), (0, 1e-9, 0))
        charges = (Dipole(100e12*2*np.pi, origin_list[0], initial_d_list[0]),
                   Dipole(100e12*2*np.pi, origin_list[1], initial_d_list[1]))
        simulation = Simulation(charges)
        simulation.run(15000, 1e-18)
        theory_delta_12, theory_gamma_12 = pc.s_dipole_theory(1e-9, 80e-9,
                                                              100e12*2*np.pi,
                                                              True)
        delta_12, gamma = pc.calculate_dipole_properties(
            charges[0], first_index=5000, print_values=True
        )
        np.testing.assert_almost_equal(gamma, theory_gamma_12+1, 1)
        np.testing.assert_almost_equal(delta_12, theory_delta_12, 1)

    def test_p_dipole_properties(self):
        origin_list = ((0, 0, 0), (80e-9, 0, 0))
        initial_d_list = ((1e-9, 0, 0), (1e-9, 0, 0))
        charges = (Dipole(100e12*2*np.pi, origin_list[0], initial_d_list[0]),
                   Dipole(100e12*2*np.pi, origin_list[1], initial_d_list[1]))
        simulation = Simulation(charges)
        simulation.run(15000, 1e-18)
        theory_delta_12, theory_gamma_12 = pc.p_dipole_theory(1e-9, 80e-9,
                                                              100e12*2*np.pi,
                                                              True)
        delta_12, gamma = pc.calculate_dipole_properties(
            charges[0], first_index=5000
        )

        self.assertTrue(np.all(charges[0].get_origin_position() == 0))
        np.testing.assert_almost_equal(gamma, theory_gamma_12+1, 1)
        np.testing.assert_almost_equal(delta_12, theory_delta_12, 1)

    def test_different_masses(self):
        timesteps = 10000
        source1 = Dipole(100e12*2*np.pi, (0, 0, 0), (1e-9, 0, 0), m=m_e*2)
        simulation1 = Simulation(source1)
        simulation1.run(timesteps, 1e-18)
        source2 = Dipole(100e12*2*np.pi, (0, 0, 0),
                         (1e-9, 0, 0), m=(m_e, 1e18))
        simulation2 = Simulation(source2)
        simulation2.run(timesteps, 1e-18)

        np.testing.assert_almost_equal(source1.m_eff, source2.m_eff)
        np.testing.assert_almost_equal(source1.get_kinetic_energy(),
                                       source2.get_kinetic_energy())
        x1 = source1.charge_pair[0].xpos(np.arange(timesteps-1)*1e-18)
        x2 = source2.charge_pair[0].xpos(np.arange(timesteps-1)*1e-18)
        np.testing.assert_almost_equal(x1*2, x2)


if __name__ == '__main__':
    unittest.main()
