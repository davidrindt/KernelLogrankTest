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
diabetic = pd.read_csv('../../data/diabetic.csv')
diabetic_no_trt = diabetic[diabetic.trt == 0]
diabetic_trt = diabetic[diabetic.trt == 1]


# Treatment group. Dependence risk with time
print('trt. Risk vs time')

grp = diabetic_trt
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
survival_scatter_plot(x=x, z=z, d=d)

v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)


# Treatment group and left eye. Dependence risk with time

print('trt, left. Risk vs time')

grp = diabetic_trt[diabetic_trt.eye == 'left']
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)

# Treatment group and right eye. Dependence risk with time

print('trt, right. Risk vs time')
grp = diabetic_trt[diabetic_trt.eye == 'right']
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)


# No treatment group. Dependence risk with time
print('No trt. Risk vs time')

grp = diabetic_no_trt
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values

v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)


# No treatment group. Dependence risk with time
print('No trt left. Risk vs time')
grp = diabetic_no_trt[diabetic_no_trt.eye == 'left']
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)


# No treatment group. Dependence risk with time
print('No trt right. Risk vs time')
grp = diabetic_no_trt[diabetic_no_trt.eye == 'right']
x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)


# No treatment group. Dependence age with time
print('No trt. Age vs time')

grp = diabetic_no_trt
x, z, d = grp.age.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)

# Treatment group. Dependence age with time
print('Trt. Age vs time')

grp = diabetic_trt
x, z, d = grp.age.values[:,None], grp.time.values, grp.status.values


v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)



