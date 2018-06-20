import numpy as np
import spectral_estimators as se


def q_spice(y, B, M, q):
    """

    :param y: Signal of interest
    :param B: Dictionary matrix
    :param M: Number of dictionary elements
    :param q: q-norm
    :return: Frequency estimate

    TODO: Fix mirroring bug on p
    """
    N = y.shape[0]
    p = np.array([np.power(np.abs(np.dot(b, y)), 2) / np.power(np.linalg.norm(b), 4) for b in B.T])
    sigma = np.abs(y)

    A = np.hstack([B, np.eye(N)])
    L = 10
    for i in range(L):
        P = np.diag(np.vstack([p, sigma]).reshape(-1))
        R = np.dot(np.dot(A, P), A.T)
        Q = np.dot(np.dot(P, A.T), np.linalg.inv(R))
        beta = np.dot(Q, y)
        W = np.diag(np.power(se.vec_norm(A), 2)) / np.power(np.linalg.norm(y), 2)
        lam = np.linalg.norm(np.dot(np.sqrt(W[:M, :M]), beta[:M]), 1) + np.power(
            se.q_norm(np.dot(np.sqrt(W[M:, M:]), beta[M:]), 2 * q / (q + 1)), ((q - 1) / (q + 1)))
        p = np.array([np.abs(beta[k]) / (np.sqrt(W[k, k]) * np.sqrt(lam)) for k in range(M)])
        sigma = np.array([np.power(np.abs(beta[M + l]), 2 / (q + 1)) * np.power(
            se.q_norm(np.dot(np.sqrt(W[M:, M:]), beta[M:]), 2 * q / (q + 1)), (q - 1) / (q + 1)) / np.power(
            W[M + l, M + l], (q / (q + 1))) * np.sqrt(lam) for l in range(N)])
    return p[::-1, :]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    q = 1
    M, N = 200, 64

    ff = np.arange(M) / M - 0.5
    ff = ff.reshape(-1, 1)
    B = se.get_fourier_matrix(N, ff)
    y = se.complex_sinusoid(0.3, N) + se.complex_sinusoid(0.2, N)

    p = q_spice(y, B, M, q)
    plt.stem(ff, np.abs(p))
    plt.show()
