#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:45:47 2019

@author: david.rindt
"""
import sys

sys.path.append('../utils')
import numpy as np
import pandas as pd
from get_kernel_matrix import get_total_kernel_matrix


def wild_bootstrap_test_logrank_covariates(x,
                                           z,
                                           d,
                                           kernels_x,
                                           kernel_z,
                                           kernel_parameters_x=None,
                                           kernel_parameters_z=None,
                                           seed=1,
                                           num_bootstrap_statistics=1999,
                                           fast_computation=True):

    local_state = np.random.RandomState(seed)
    n = np.shape(x)[0]

    # Sort the data in order of increasing time.
    indices = np.arange(n)
    z, list2 = (np.array((t)) for t in zip(*sorted(zip(z, indices))))
    d = d[list2]
    x = x[list2]

    # Define Y_matrix[i,:] to be the vector of indicators who are at risk at the i-th event time.
    Y_matrix = np.triu(np.ones(n))
    # Define Y[i] count the number of individuals at risk at the i-th event time.
    Y = n - np.arange(n)

    # Define A[i,:] to be a normalized (each row sums to 1) indicator of being at risk at time i.
    scale_by_Y = np.diag(1 / Y)
    A = np.matmul(scale_by_Y, Y_matrix)

    # Define censoring_matrix[i,j] to be d[i]d[j]
    censoring_matrix = np.outer(d, d)

    # Subtract A from the identity matrix
    I_minus_A = np.identity(n) - A

    Kx = get_total_kernel_matrix(x, kernels_x, kernel_parameters=kernel_parameters_x, d=d)
    Kz = get_total_kernel_matrix(z[:, None], kernel_z, kernel_parameters=kernel_parameters_z, d=d)

    # Define Lz to be the kernel matrix on Z, with elementwise multiplication of the censoring matrix.
    Lz = np.multiply(Kz, censoring_matrix)

    # Define the first_product matrix that we can re-use for computation in the wilde bootstrap.

    if fast_computation == True:
        M1 = (Kx - np.divide(np.flip(np.cumsum(np.flip(Kx, axis=0), axis=0), axis=0), Y[:, None]))  # (I-A)Kx
        first_product = M1 - np.divide(np.flip(np.cumsum(np.flip(M1, axis=1), axis=1), axis=1), Y[None, :])
    else:
        first_product = np.matmul(np.matmul(I_minus_A, Kx), np.transpose(I_minus_A))

    original_statistic = np.sum(np.multiply(first_product, Lz))

    statistic_list = np.zeros(num_bootstrap_statistics + 1)
    statistic_list[0] = original_statistic

    for b in range(num_bootstrap_statistics):
        W = local_state.binomial(1, 1 / 2, size=n) * 2 - 1
        WM = np.outer(W, W)
        bootstrapLz = np.multiply(WM, Lz)
        multmatrix = np.multiply(first_product, bootstrapLz)
        bootstrap_statistic = np.sum(multmatrix)
        statistic_list[b + 1] = bootstrap_statistic

    vec = pd.Series(statistic_list)
    ranks = vec.sample(frac=1).rank(method='first', ascending=False)
    rank = ranks[0]
    p = (rank - 1) / (num_bootstrap_statistics + 1)
    return original_statistic, p


# if __name__ == '__main__':

    # Generate some data
    # local_state = np.random.RandomState(1)
    # n = 200
    # dim = 10
    # x = local_state.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=n)
    # c = np.zeros(n)
    # rowsum = np.sum(x, axis=1)
    # for v in range(n):
    #     c[v] = local_state.exponential(np.exp((rowsum[v] / 8)))
    # t = local_state.exponential(.6, size=n)
    # d = np.int64(c > t)
    # z = np.minimum(t, c)
    #
    # # Run the test
    # print(wild_bootstrap_test_logrank_covariates(x, z, d, ['gau'], 'gau'))
    # print(wild_bootstrap_test_logrank_covariates(x, z, d, ['gau'], 'gau', fast_computation=True))
