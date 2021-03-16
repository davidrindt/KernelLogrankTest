#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:14:34 2021

@author: rindt
"""
import sys
import numpy as np
from FisherInformation import inv_inf_matrix
from kerpy import LinearKernel
from kerpy import GaussianKernel
from kerpy import PolynomialKernel
from scipy.spatial.distance import pdist, squareform

from scipy.spatial.distance import pdist, squareform

def get_total_kernel_matrix(X, kernels, kernel_parameters=None, d=None):
    '''
    NOTE: X and d need to be sorted by event time Z
    '''
    X = ( X - X.mean(axis=0) ) / X.std(axis=0)
    n, p = X.shape

    if (type(kernels) == list):
        if kernel_parameters is None:
            kernel_parameters = [None for _ in range(p)]
        Kx = np.ones((n, n))
        for i, kernel, parameter in zip(np.arange(p), kernels, kernel_parameters):
            m = get_kernel_matrix(X[:, i][:, None], kernel=kernel, parameter=parameter)
            Kx *= m

    elif type(kernels) is str:
        Kx = get_kernel_matrix(X, kernels, kernel_parameters)

    else:
        Kx = 1
        print("Error")
    return Kx

def get_kernel_matrix(X, kernel, parameter=None):
    '''
    NOTE: X and d need to be sorted by event time Z
    '''
    X = ( X - X.mean(axis=0) ) / X.std(axis=0)
    n, p = X.shape

    if kernel == 'linfis':
        inverse_inf_matrix = inv_inf_matrix(sorted_X=X, sorted_d=d)
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
    elif kernel == 'pol':
        k = PolynomialKernel.PolynomialKernel(degree=parameter)
        Kx = k.kernel(X)
    elif kernel == 'bin':
        f = lambda a, b: 0 if a == b else 0.6
        Kx = 1.4 * np.ones((n, n)) - squareform(pdist(X, f))
    elif kernel == 'con':
        Kx = np.ones((n, n))
    else:
        print('Error, kernel not recognized')
    return Kx


# if __name__ == '__main__':
    # n = 6
    # X = np.random.binomial(1, 0.5, size=n)
    # print(X)
    # X = X[:, None]
    # print('X', X)
    # print(' the kernel metrix', get_kernel_matrix(X, 'bin'))
    #
    # X = np.zeros((n,3))
    # X[:, 0] = np.random.binomial(1, 0.5, size=n)
    # X[:, 1:] = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2), size=n)
    # kernels = ['bin', 'gau', 'gau']
    # params = None
    # Kx = get_total_kernel_matrix(X, kernels, params)
    # print(Kx)
    #
    # X = np.random.multivariate_normal(np.zeros(2), cov=np.identity(2), size=n)
    # kernels = ['gau', 'gau']
    # params = [10., 10.]
    # Kx = get_total_kernel_matrix(X, kernels, params)
    # print(Kx)
    #
    
    
    
    
    
    
    
    
    
    
        #Define the kernel matrix on X
# =============================================================================
#     if kernel_x=='linfis':
#         inverse_inf_matrix=inv_inf_matrix(sorted_X=x, sorted_z=z,sorted_d=d)
#         Kx=np.matmul(np.matmul(x,inverse_inf_matrix), np.transpose(x))
#     elif kernel_x=='lin':
#         k=LinearKernel.LinearKernel()
#         Kx=k.kernel(x)   
#     elif kernel_x=='gau':
#         k=GaussianKernel.GaussianKernel()
#         k.width = k.get_sigma_median_heuristic(x)
#         Kx=k.kernel(x)
#     elif kernel_x=='pol':
#         k=PolynomialKernel.PolynomialKernel(degree=2)
#         Kx=k.kernel(x)
#     elif kernel_x=='gaufis':
#         inverse_inf_matrix=inv_inf_matrix(sorted_X=x, sorted_z=z, sorted_d=d)
#         #We first compute an matrix M such that matrix_M[i,j]=x[i]^TVx[i]+x[j]^TVx[j]-2x[i]^TVx[j]=matrix_A[i,j]+matrix_A[j,i]-2matrix_C[i,j]
#         matrix_C=np.matmul(np.matmul(x, inverse_inf_matrix), np.transpose(x))
#         matrix_A=np.matmul(np.ones((n, n)), np.diag(np.diag(matrix_C)))
#         matrix_M=matrix_A+np.transpose(matrix_A) - 2 * matrix_C
#         sigma=np.median(matrix_M)
#         Kx=np.exp(-matrix_M/ (2 * sigma))
#     else:
#         print('kernel_x: choose from linfis, lin, gau, pol, gaufis')
# 
#     #Define the kernel matrix on Z
#     if kernel_z=='gau':
#         l=GaussianKernel.GaussianKernel()
#         l.width = l.get_sigma_median_heuristic(z[:,np.newaxis])
#         Kz=l.kernel(z[:,np.newaxis])
#     elif kernel_z=='con':
#         Kz=np.ones((n,n))
#     elif kernel_z=='pol':
#         l=PolynomialKernel.PolynomialKernel(degree=2)
#         Kz=k.kernel(z[:,np.newaxis])
#     else:
#         print('kernel_z: choose from gau, con')
# =============================================================================
