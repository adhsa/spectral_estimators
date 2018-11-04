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

    spectrum= np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        a = A[:, i].reshape(-1, 1)
        spectrum[i] = np.real(np.squeeze(np.dot(np.conj(a).T, np.dot(R, a))))
    return spectrum
