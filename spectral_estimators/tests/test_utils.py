import unittest
import numpy as np

import spectral_estimators as se


class TestComplexSinusoid(unittest.TestCase):
    def test_zero(self):
        for n in range(1, 100):
            t = np.random.randn(n)
            y = se.utils.complex_sinusoid(0, t)

            np.testing.assert_allclose(np.ones(n), np.real(y))
            np.testing.assert_allclose(np.zeros(n), np.imag(y))

    def test_unit_circle(self):
        self.assertAlmostEqual(-1, se.utils.complex_sinusoid(np.pi, 1))
        self.assertAlmostEqual(1, se.utils.complex_sinusoid(0, 1))
        self.assertAlmostEqual(1j, se.utils.complex_sinusoid(np.pi / 2, 1))
        self.assertAlmostEqual(-1j, se.utils.complex_sinusoid(-np.pi / 2, 1))


class TestSteeringVector(unittest.TestCase):
    def test_forward(self):
        for n in range(1, 100):
            a = se.utils.steering_vector(0, n)
            np.testing.assert_allclose(np.ones(n), a)

    def test_left_side(self):
        for n in range(1, 100):
            a = se.utils.steering_vector(90, n)
            for i in range(n):
                if i % 2 == 0:
                    self.assertAlmostEqual(1, a[i])
                else:
                    self.assertAlmostEqual(-1, a[i])

    def test_right_side(self):
        for n in range(1, 100):
            a = se.utils.steering_vector(-90, n)
            for i in range(n):
                if i % 2 == 0:
                    self.assertAlmostEqual(1, a[i])
                else:
                    self.assertAlmostEqual(-1, a[i])

    def test_right_quarter(self):
        a_true = np.array([1.000000000000000 + 1j * 0.000000000000000,
                           -0.605699867078813 - 1j * 0.795693201567481,
                           -0.266255342041416 + 1j * 0.963902532849877,
                           0.928241517645832 - 1j * 0.371978070480724,
                           -0.858216185668818 - 1j * 0.513288397157062])
        a = se.utils.steering_vector(45, 5)
        np.testing.assert_allclose(a_true, a)

    def test_left_quarter(self):
        a_true = np.array([1.000000000000000 + 1j * 0.000000000000000,
                           -0.605699867078813 + 1j * 0.795693201567481,
                           -0.266255342041416 - 1j * 0.963902532849877,
                           0.928241517645832 + 1j * 0.371978070480724,
                           -0.858216185668818 + 1j * 0.513288397157062])
        a = se.utils.steering_vector(-45, 5)
        np.testing.assert_allclose(a_true, a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
