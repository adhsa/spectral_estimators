import numpy as np


def get_fourier_matrix(N, ff):
    """

    :param N: Number of samples
    :param ff: Frequency grid
    :return: Fourier matrix
    """
    return np.exp(1j * 2 * np.pi * np.outer(np.arange(1, N + 1).reshape(-1, 1), ff))


def complex_sinusoid(f, N, amp=1, sigma=1):
    """ Generate a complex sinusoid

    :param f: Signal frequency
    :param amp: Signal amplitude
    :param sigma: Noise variance
    :param N: Number of samples
    :return: Complex sinusoid
    """
    y = amp * np.exp(1j * f * 2 * np.pi * np.arange(1, N + 1).reshape(-1, 1) + 1j * np.pi * np.random.rand())
    w = sigma * (np.random.randn(N).reshape(-1, 1) + 1j * np.random.randn(N).reshape(-1, 1)) / np.sqrt(2)
    return y + w


def vec_norm(A):
    """

    :param A:
    :return:
    """
    return np.array([np.linalg.norm(a) for a in A.T])


def q_norm(x, p):
    """

    :param x:
    :param p:
    :return:
    """
    return np.power(np.sum(np.power(np.abs(x), p)), 1 / p)
