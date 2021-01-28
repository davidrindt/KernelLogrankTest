#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 12:14:34 2021

@author: rindt
"""

import numpy as np
from FisherInformation import inv_inf_matrix
from kerpy import LinearKernel
from kerpy import GaussianKernel
from kerpy import PolynomialKernel
from scipy.spatial.distance import pdist, squareform

def get_kernel_matrix(X, kernel, bandwidth=None, d=None):
    '''
    NOTE: X and d need to be sorted by event time Z
    '''
    n = X.shape[0]
    if kernel=='linfis':
        inverse_inf_matrix = inv_inf_matrix(sorted_X=X, sorted_d=d)
        Kx= np.matmul(np.matmul(X, inverse_inf_matrix), np.transpose(X))
    elif kernel == 'lin':
        k = LinearKernel.LinearKernel()
        Kx = k.kernel(X)   
    elif kernel == 'gau':
        k = GaussianKernel.GaussianKernel()
        k.width = k.get_sigma_median_heuristic(X)
        Kx = k.kernel(X)
    elif kernel == 'pol':
        k = PolynomialKernel.PolynomialKernel(degree=2)
        Kx = k.kernel(X)
    elif kernel == 'euc':
        v = pdist(X, metric= 'euclidean')
        Kx = 1 - squareform(v)    
    elif kernel == 'con':
        Kx = np.ones((n,n))
    else: 
        print('Error, kernel not recognized')
    return Kx
    

if __name__ == '__main__':
    X = np.random.binomial(1, 0.5, size=10)
    X = X[:, None]
    print(get_kernel_matrix(X, 'euc'))
    
    
    
    
    
    
    
    
    
    
    
    
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
