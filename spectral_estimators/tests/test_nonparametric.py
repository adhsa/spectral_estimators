import unittest
import numpy as np

from spectral_estimators.covariance import covariance_matrix
from spectral_estimators.nonparametric import beamformer
from spectral_estimators.utils import steering_matrix, single_radar_target


class TestBeamformer(unittest.TestCase):

    def test_single_peak(self):
        """ This method sweeps through several peaks and tests if the beamformer gives a correct estimate.

        TODO: Make this into a decorator

        """

        n_sensors = 12
        theta_grid = np.linspace(-90, 90, 720)
        A = steering_matrix(theta_grid, n_sensors)

        thetas = np.linspace(-80, 80, 20)
        for theta in thetas:
            data = single_radar_target(theta, n_sensors)
            data += 0.1 * (np.random.randn(*data.shape) + 1j * np.random.randn(*data.shape)) / np.sqrt(2)

            R = covariance_matrix(data)
            spectrum = beamformer(R, A)

            ind = np.argmax(spectrum)
            estimated = theta_grid[ind]
            error = np.linalg.norm(theta - estimated)
            self.assertGreater(0.5, error)
