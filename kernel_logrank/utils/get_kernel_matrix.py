import numpy as np
from kernel_logrank.utils.fisher_information import inv_inf_matrix
from kerpy import LinearKernel
from kerpy import GaussianKernel
from scipy.spatial.distance import pdist, squareform


def get_total_kernel_matrix(X, kernels, kernel_parameters=None, d=None):
    '''
    NOTE: X and d need to be sorted by event time Z
    '''
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    n, p = X.shape
    if type(kernels) == list:
        assert len(kernels) == p
        if kernel_parameters is None:
            kernel_parameters = [None for _ in range(p)]
        Kx = np.ones((n, n))
        for i, kernel, parameter in zip(np.arange(p), kernels, kernel_parameters):
            m = get_kernel_matrix(X[:, i][:, None], d, kernel=kernel, parameter=parameter)
            Kx *= m

    elif type(kernels) is str:
        Kx = get_kernel_matrix(X, d, kernels, kernel_parameters)

    else:
        Kx = 1
        print("Error")

    return Kx


def get_kernel_matrix(X, d, kernel, parameter=None):
    '''
    @param X: 2d numpy array
    @param kernel:
    @param parameter:
    @return:
    '''
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    n, p = X.shape

    if kernel == 'linfis':
        inverse_inf_matrix = inv_inf_matrix(X=X, d=d)
        Kx = np.matmul(np.matmul(X, inverse_inf_matrix), np.transpose(X))

    elif kernel == 'lin':
        k = LinearKernel.LinearKernel()
        Kx = k.kernel(X)

    elif kernel == 'gau':
        k = GaussianKernel.GaussianKernel()
        if parameter is None:
            k.width = k.get_sigma_median_heuristic(X)
        else:
            k.width = parameter
        Kx = k.kernel(X)

    elif kernel == 'bin':
        f = lambda a, b: 0 if a == b else 0.6
        Kx = 1.4 * np.ones((n, n)) - squareform(pdist(X, f))

    elif kernel == 'con':
        Kx = np.ones((n, n))

    else:
        print('WARNING, kernel not recognized')

    return Kx


if __name__ == '__main__':
    n = 6

    # Try out binomial kernel
    X = np.random.binomial(1, 0.5, size=n)
    X = X[:, None]
    print(f'X, {X}')
    Kx = get_kernel_matrix(X, 'bin')
    print(f' the kernel matrix is {Kx}')

    # Try out a list kernel
    X = np.zeros((n, 3))
    X[:, 0] = np.random.binomial(1, 0.5, size=n)
    X[:, 1:] = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2), size=n)
    kernels = ['bin', 'gau', 'gau']
    params = None
    Kx = get_total_kernel_matrix(X, kernels, params)
    print(f' the kernel matrix is {Kx}')

    # Try out a list kernel of Gaussians
    X = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2), size=n)
    kernels = ['gau', 'gau']
    params = [10., 10.]
    Kx = get_total_kernel_matrix(X, kernels, params)
    print(f' the kernel matrix is {Kx}')
