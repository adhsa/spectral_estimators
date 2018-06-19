import numpy as np


def get_fourier_matrix(N, ff):
    A = np.exp(1j * 2 * np.pi * np.outer(np.arange(1, N + 1).reshape(-1, 1), ff))
    return A


def get_some_data(f, N):
    y = np.exp(1j * f * 2 * np.pi * np.arange(1, N + 1).reshape(-1, 1) + 1j * np.pi * np.random.rand())
    return y + (np.random.randn(N).reshape(-1, 1) + 1j * np.random.randn(N).reshape(-1, 1)) / np.sqrt(2)


def vec_norm(A):
    norm = np.array([np.linalg.norm(a) for a in A.T])
    return norm


def qnorm(x, p):
    out = np.power(np.sum(np.power(np.abs(x), p)), 1 / p)
    return out


def q_spice(y, B, M, q):
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
        W = np.diag(np.power(vec_norm(A), 2)) / np.power(np.linalg.norm(y), 2)
        lam = np.linalg.norm(np.dot(np.sqrt(W[:M, :M]), beta[:M]), 1) + np.power(
            qnorm(np.dot(np.sqrt(W[M:, M:]), beta[M:]), 2 * q / (q + 1)), ((q - 1) / (q + 1)))
        p = np.array([np.abs(beta[k]) / (np.sqrt(W[k, k]) * np.sqrt(lam)) for k in range(M)])
        sigma = np.array([np.power(np.abs(beta[M + l]), 2 / (q + 1)) *
                          np.power(qnorm(np.dot(np.sqrt(W[M:, M:]), beta[M:]), 2 * q / (q + 1)), (q - 1) / (q + 1))
                          / np.power(W[M + l, M + l], (q / (q + 1))) * np.sqrt(lam) for l in range(N)])
    return p[::-1, :]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    q = 1.5
    M, N = 500, 64

    ff = np.arange(M) / M - 0.5
    ff = ff.reshape(-1, 1)
    B = get_fourier_matrix(N, ff)
    y = get_some_data(0.3, N)

    p = q_spice(y, B, M, q)
    plt.stem(ff, np.abs(p))
    plt.show()



