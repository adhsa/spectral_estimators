import numpy as np


def covariance_matrix(data, fb=False, spsmooth=0, method='scm'):
    """

    Parameters
    ----------
    data : np.ndarray
        The data used to create the covariance matrix. Is of shape (n_samples, n_snapshots)

    fb : bool
        If true, uses forward-backward averaging

    spsmooth : int
        Number of subarrays to use for spatial smoothing.

    method : str
        The method used to estimate the covariance matrix. ('scm', 'ncm', 'pfm')

    Returns
    -------

    """

    R = np.dot(data, np.conj(data).T) / data.shape[1]
    return R
