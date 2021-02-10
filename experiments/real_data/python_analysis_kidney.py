#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:54:29 2021

@author: rindt
"""

import sys
sys.path.append('../../utils')
sys.path.append('../../data')
sys.path.append('../../tests')

import numpy as np
import pandas as pd
from survival_scatter_plot import survival_scatter_plot
import wild_bootstrap_LR 
kidney = pd.read_csv('../../data/kidney.csv')
kidney_male = kidney[kidney.sex == 1]
kidney_female = kidney[kidney.sex == 2]

print(kidney.head(10))


# Test if age has relationship with time in each of the subpopulations
x_female, z_female, d_female = kidney_female.age.values[:,None], kidney_female.time.values, kidney_female.status.values
survival_scatter_plot(x=x_female.flatten(), d=d_female, z=z_female)

x_male, z_male, d_male = kidney_male.age.values[:,None], kidney_male.time.values, kidney_male.status.values
survival_scatter_plot(x=x_male.flatten(), d=d_male, z=z_male)

v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)



# Overall test if time depends on (age, sex, frail).

X = kidney[['age', 'sex', 'frail']]
print(X)
X = np.array(X)
print(X)
z = kidney.time.values
d = kidney.status.values

B = int(10e6)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='lin', kernel_z='con', num_bootstrap_statistics=B)
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='con', num_bootstrap_statistics=B)
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='gau', num_bootstrap_statistics=B)
print('gua, gau', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='con', num_bootstrap_statistics=B)
print('linfis, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='gau', num_bootstrap_statistics=B)
print('linfis, gau', p, v)
