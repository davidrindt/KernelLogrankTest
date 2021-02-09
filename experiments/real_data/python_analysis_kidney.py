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

kidney.head(10)

x_female, z_female, d_female = kidney_female.age.values[:,None], kidney_female.time.values, kidney_female.status.values
survival_scatter_plot(x=x_female.flatten(), d=d_female, z=z_female)
plt.show()

x_male, z_male, d_male = kidney_male.age.values[:,None], kidney_male.time.values, kidney_male.status.values
survival_scatter_plot(x=x_male.flatten(), d=d_male, z=z_male)
plt.show()

v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='lin', kernel_z='con')
print('lin, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='gau', kernel_z='con')
print('gau, con', p, v)
v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x_female, z=z_female, d=d_female, kernel_x='gau', kernel_z='gau')
print('gua, gau', p, v)