import numpy as np


def beamformer(R, A):
    """ This function operates the standard beamformer.

    Parameters
    ----------
    R : np.ndarray
        The covariance matrix

    A : np.ndarray
        The steering vector matrix. Is of shape (n_sensors, n_vectors)

    Returns
    -------
    np.ndarray
        The estimated power spectrum.

    """

    pass