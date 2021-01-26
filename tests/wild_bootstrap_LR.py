#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:45:47 2019

@author: david.rindt
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from FisherInformation import inv_inf_matrix
from kerpy import LinearKernel
from kerpy import GaussianKernel
from kerpy import PolynomialKernel



def wild_bootstrap_test_logrank_covariates(x,
                                           z,
                                           d,
                                           kernel_x,
                                           kernel_z,
                                           alpha=0.05,
                                           seed=1,
                                           num_bootstrap_statistics=1999,
                                           print_score=False,
                                           fast_computation=True):
    
    local_state=np.random.RandomState(seed)
    n=np.shape(x)[0]

    #Sort the data in order of increasing time.
    indices=np.arange(n)
    z, list2= (np.array((t)) for t in zip(*sorted(zip(z,indices))))
    d=d[list2]
    x=x[list2]
    
    #Define Y_matrix[i,:] to be the vector of indicators who are at risk at the i-th event time.
    Y_matrix=np.triu(np.ones(n))
    #Define Y[i] count the number of individuals at risk at the i-th event time. 
    Y=n-np.arange(n)
    
    #Define A[i,:] to be a normalized (each row sums to 1) indicator of being at risk at time i. (note this is the transpose of A in our paper).
    scale_by_Y=np.diag(1/Y)
    A=np.matmul(scale_by_Y,Y_matrix)

    #Define censoring_matrix[i,j] to be d[i]d[j]
    censoring_matrix=np.outer(d,d)
    
    #Subtract A from the identity matrix
    I_minus_A=np.identity(n)-A
    

    #Define the kernel matrix on X
    if kernel_x=='linfis':
        inverse_inf_matrix=inv_inf_matrix(sorted_X=x,sorted_z=z,sorted_d=d,print_score=print_score)
        Kx=np.matmul(np.matmul(x,inverse_inf_matrix),np.transpose(x))
    elif kernel_x=='lin':
        k=LinearKernel.LinearKernel()
        Kx=k.kernel(x)   
    elif kernel_x=='gau':
        k=GaussianKernel.GaussianKernel()
        k.width = k.get_sigma_median_heuristic(x)
        Kx=k.kernel(x)
    elif kernel_x=='pol':
        k=PolynomialKernel.PolynomialKernel(degree=2)
        Kx=k.kernel(x)
    elif kernel_x=='gaufis':
        inverse_inf_matrix=inv_inf_matrix(sorted_X=x,sorted_z=z,sorted_d=d)
        #We first compute an matrix M such that matrix_M[i,j]=x[i]^TVx[i]+x[j]^TVx[j]-2x[i]^TVx[j]=matrix_A[i,j]+matrix_A[j,i]-2matrix_C[i,j]
        matrix_C=np.matmul(np.matmul(x,inverse_inf_matrix),np.transpose(x))
        matrix_A=np.matmul(np.ones((n,n)),np.diag(np.diag(matrix_C)))
        matrix_M=matrix_A+np.transpose(matrix_A)-2*matrix_C
        sigma=np.median(matrix_M)
        Kx=np.exp(-matrix_M/(2*sigma))
    else:
        print('kernel_x: choose from linfis, lin, gau, pol, gaufis')

    #Define the kernel matrix on Z
    if kernel_z=='gau':
        l=GaussianKernel.GaussianKernel()
        l.width = l.get_sigma_median_heuristic(z[:,np.newaxis])
        Kz=l.kernel(z[:,np.newaxis])
    elif kernel_z=='con':
        Kz=np.ones((n,n))
    elif kernel_z=='pol':
        l=PolynomialKernel.PolynomialKernel(degree=2)
        Kz=k.kernel(z[:,np.newaxis])
    else:
        print('kernel_z: choose from gau, con')

    #Define Lz to be the kernel matrix on Z, with elementwise multiplication of the censoring matrix.
    Lz=np.multiply(Kz,censoring_matrix)   
    
    #Define the first_product matrix that we can re-use for computation in the wilde bootstrap.
    
    if fast_computation == True:
        M1=(Kx-np.divide(np.flip(np.cumsum(np.flip(Kx,axis=0),axis=0),axis=0),Y[:,None])) # (I-A)Kx
        first_product= M1-np.divide(np.flip(np.cumsum(np.flip(M1,axis=1),axis=1),axis=1),Y[None,:])
    else:
        first_product=np.matmul(np.matmul(I_minus_A,Kx),np.transpose(I_minus_A))
        
        
    original_statistic=np.sum(np.multiply(first_product,Lz))
    if print_score:
        print('the lr statistic is',original_statistic)

    statistic_list=np.zeros(num_bootstrap_statistics+1)
    statistic_list[0]=original_statistic

    for b in range(num_bootstrap_statistics):
        W=local_state.binomial(1,1/2,size=n)*2-1
        WM=np.outer(W,W)
        bootstrapLz=np.multiply(WM,Lz)
        multmatrix=np.multiply(first_product,bootstrapLz)
        bootstrap_statistic=np.sum(multmatrix)
        statistic_list[b+1]=bootstrap_statistic
        

    vec=pd.Series(statistic_list)
    vec=vec.sample(frac=1).rank(method='first')
    k=vec[0]
    return((num_bootstrap_statistics-k+2)/(num_bootstrap_statistics+1))
    
    
if __name__ == '__main__':
    print('hey')