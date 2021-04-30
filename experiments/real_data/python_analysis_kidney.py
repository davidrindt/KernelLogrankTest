#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:54:29 2021

@author: rindt
"""

import sys
sys.path.append('../../kernel_logrank/utils')
sys.path.append('../../kernel_logrank/data')
sys.path.append('../../kernel_logrank/tests')

import numpy as np
import pandas as pd
from survival_scatter_plot import survival_scatter_plot
import wild_bootstrap_LR 
from CPH_test import CPH_test
kidney = pd.read_csv('../../kernel_logrank/data/kidney.csv')
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

B = int(10e3)
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

#
## Overall test if time depends on (age, sex, frail).
#
#X = kidney[['age', 'sex', 'frail']]
#print(X)
#X = np.array(X)
#print(X)
#z = kidney.time.values
#d = kidney.status.values
#
#B = int(10e3)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='lin', kernel_z='con', num_bootstrap_statistics=B)
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='con', num_bootstrap_statistics=B)
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='gau', num_bootstrap_statistics=B)
#print('gua, gau', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='con', num_bootstrap_statistics=B)
#print('linfis, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='gau', num_bootstrap_statistics=B)
#print('linfis, gau', p, v)
#



# Subsample rows
s_kidney = kidney[['time', 'status', 'age', 'sex', 'frail']]
s_kidney = s_kidney.sample(50)

X = np.array(s_kidney[['age', 'sex', 'frail']])
z = s_kidney.time.values
d = s_kidney.status.values
print(s_kidney)
print(X)

B = int(10e5)
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
print(X)

print('CPH test', CPH_test(X, z, d))
