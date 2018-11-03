import unittest
import numpy as np

from spectral_estimators.covariance import covariance_matrix


class TestCovarianceMatrix(unittest.TestCase):

    def test_white(self):
        for n in range(1, 20):
            y = np.random.randn(n, 500000)
            R = covariance_matrix(y)
            np.testing.assert_allclose(np.eye(n), R, atol=1e-2)

    def test_sinusoid(self):
        t = np.linspace(0, 4 * np.pi, 100)
        y = np.array([np.sin(t) * np.exp(1j * 2 * np.pi * np.random.rand(1)) for _ in range(200)]).T
        R = covariance_matrix(y)

        U, E, V = np.linalg.svd(R)
        print(E)

