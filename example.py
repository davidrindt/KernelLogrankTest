# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:13:32 2018

@author: David
"""


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from CPH_test import CPH_test
from wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from opthsic import opt_hsic
import pickle
import sys
from scipy.linalg import expm

#This file consists of two examples, where we generate data, and run various test: the CPH likelihood ratio test, the kernel logrank test with different kernels and the OPTHSIC test.


#np.random.seed(1)
#local_state=np.random.RandomState(1)
local_state=np.random.RandomState(np.random.randint(10000000))



### EXAMPLE 1
#Generate the data:
n=200
x=local_state.uniform(low=-1,high=1,size=n)
t=np.zeros(n)
for v in range(n):
    t[v]=local_state.exponential(np.exp(x[v]/3))
c=local_state.exponential(scale=1.5,size=n)
x=x[:,np.newaxis]
d= np.int64(c > t)
z=np.minimum(t,c)

### Call the tests:
print('CPH p value:',CPH_test(x=x,z=z,d=d,alpha=0.05))

print('Wild bootstrap Kernel Logrank p value (gaussian kernel x, constant kernel y):',wild_bootstrap_test_logrank_covariates(x=x,z=z,d=d,seed=1,kernel_x='gau',kernel_z='con'))

print('Wild bootstrap Kernel Logrank p value (linear kernel x, constant kernel y):',wild_bootstrap_test_logrank_covariates(x=x,z=z,d=d,seed=1,kernel_x='lin',kernel_z='con'))

print('Optimal transport HSIC p value:', opt_hsic(x=x,z=z,d=d,seed=1))

###EXAMPLE 2
dim=10
x=local_state.multivariate_normal(mean=np.zeros(dim),cov=np.identity(dim),size=n)
c=np.zeros(n)
rowsum=np.sum(x,axis=1)
for v in range(n):
    c[v]=local_state.exponential(np.exp((rowsum[v]/8)))
t=local_state.exponential(.6,size=n)
d= np.int64(c > t)
z=np.minimum(t,c)


#Call the tests:
print('CPH p value:',CPH_test(x=x,z=z,d=d,alpha=0.05))

print('Wild bootstrap Kernel Logrank p value (gaussian kernel x, constant kernel y):',wild_bootstrap_test_logrank_covariates(x=x,z=z,d=d,seed=1,kernel_x='gau',kernel_z='con'))

print('Wild bootstrap Kernel Logrank p value (linear kernel x, constant kernel y):',wild_bootstrap_test_logrank_covariates(x=x,z=z,d=d,seed=1,kernel_x='lin',kernel_z='con'))

print('Optimal transport HSIC p value:', opt_hsic(x=x,z=z,d=d,seed=1))
