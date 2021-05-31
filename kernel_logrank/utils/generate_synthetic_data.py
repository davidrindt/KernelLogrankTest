import numpy as np


def generate_synthetic_data(n=400, dim=10):
    '''
    @param n: Integer. Sample size.
    @param dim: Integer. Dimension of covariate.
    @return: Dataset X, z, d, numpy arrays.
    '''

    # Generate some data
    M = np.random.normal(size=10 ** 2).reshape(10, 10)
    psd = np.matmul(M, np.transpose(M))
    dim = 10
    X = np.random.multivariate_normal(mean=np.zeros(dim), cov=psd, size=n)
    t = np.zeros(n)

    for v in range(n):
        t[v] = np.random.exponential(np.exp((((X[v, 0]) ** 2) / 8)))

    c = np.random.exponential(3, size=n)
    d = np.int64(c > t)
    z = np.minimum(t, c)

    return X, z, d
