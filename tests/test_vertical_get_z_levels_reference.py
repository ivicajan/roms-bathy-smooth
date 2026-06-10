import unittest
import numpy as np

from roms_bathy_smooth.vertical import VerticalCoords
# The tests in this file can be run in the following manner: 
# python -m unittest -v tests/test_vertical_get_z_levels_reference.py

class TestVerticalGetZLevelsReference(unittest.TestCase):
    def test_vstretching5_vtransform2_matches_reference_table_h10(self):
        vertical = VerticalCoords(
            N=40,
            theta_s=5.0,
            theta_b=4.0,
            vtransform=2,
            vstretching=5,
            hc=100.0,
        )

        # Table values provided by user for the "Z at h=10 meters" column,
        # listed from level 40 (surface) down to level 0 (bottom).
        expected_surface_to_bottom = np.array([
            0.000,
            -0.091,
            -0.189,
            -0.294,
            -0.407,
            -0.527,
            -0.654,
            -0.788,
            -0.930,
            -1.079,
            -1.236,
            -1.400,
            -1.572,
            -1.752,
            -1.939,
            -2.135,
            -2.338,
            -2.551,
            -2.771,
            -3.000,
            -3.239,
            -3.486,
            -3.743,
            -4.010,
            -4.288,
            -4.576,
            -4.875,
            -5.185,
            -5.507,
            -5.841,
            -6.186,
            -6.543,
            -6.911,
            -7.288,
            -7.672,
            -8.061,
            -8.453,
            -8.844,
            -9.233,
            -9.618,
            -10.000,
        ], dtype=float)

        h = np.array([[10.0]], dtype=float)
        mask = np.array([[1]], dtype=int)
        _, z_w = vertical.get_z_levels(h, mask)

        # Internal indexing is k=0 bottom ... k=N surface, so reverse expected.
        expected_by_k = expected_surface_to_bottom[::-1]

        np.testing.assert_allclose(
            z_w[:, 0, 0],
            expected_by_k,
            rtol=0.0,
            atol=1.0e-3,
        )

    def test_vstretching4_vtransform2_matches_reference_table_h100(self):
        vertical = VerticalCoords(
            N=42,
            theta_s=6.0,
            theta_b=0.3,
            vtransform=2,
            vstretching=4,
            hc=100.0,
        )

        # Table values provided from ROMS output "at hc" column,
        # listed from level 42 (surface) down to level 0 (bottom).
        expected_surface_to_bottom = np.array([
            0.000,
            -1.193,
            -2.393,
            -3.598,
            -4.810,
            -6.029,
            -7.255,
            -8.490,
            -9.733,
            -10.987,
            -12.252,
            -13.530,
            -14.823,
            -16.133,
            -17.461,
            -18.812,
            -20.188,
            -21.593,
            -23.031,
            -24.507,
            -26.027,
            -27.597,
            -29.225,
            -30.919,
            -32.688,
            -34.545,
            -36.501,
            -38.572,
            -40.772,
            -43.121,
            -45.638,
            -48.348,
            -51.275,
            -54.449,
            -57.900,
            -61.662,
            -65.773,
            -70.271,
            -75.197,
            -80.592,
            -86.499,
            -92.957,
            -100.000,
        ], dtype=float)

        h = np.array([[100.0]], dtype=float)
        mask = np.array([[1]], dtype=int)
        _, z_w = vertical.get_z_levels(h, mask)

        # Internal indexing is k=0 bottom ... k=N surface, so reverse expected.
        expected_by_k = expected_surface_to_bottom[::-1]

        np.testing.assert_allclose(
            z_w[:, 0, 0],
            expected_by_k,
            rtol=0.0,
            atol=1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
