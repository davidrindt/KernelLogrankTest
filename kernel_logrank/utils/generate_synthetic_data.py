import numpy as np


def generate_synthetic_data(n=200, dim=10):
    '''
    @param n: Integer. Sample size.
    @param dim: Integer. Dimension of covariate.
    @return: Dataset X, z, d, numpy arrays.
    '''

    # Generate some data
    local_state = np.random.RandomState(1)
    X = local_state.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=n)
    t = np.zeros(n)

    # Generate some dependence between the covariate and the survival time
    row_sum = np.sum(X, axis=1)
    for v in range(n):
        t[v] = local_state.exponential(np.exp((row_sum[v] / 8)))

    c = local_state.exponential(.6, size=n)
    d = np.int64(c > t)
    z = np.minimum(t, c)

    return X, z, d
