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
from CPH_test import CPH_test
import numpy as np
import pandas as pd
from utils.survival_scatter_plot import survival_scatter_plot
import wild_bootstrap_LR  as wild_bootstrap_LR
import pickle
import h5py
import get_kernel_matrix

np.random.seed(1)
B = 5000

# Load data

colon = pd.read_csv('../../data/colon')
colon = colon[colon.etype == 2]
covariates = ['age', 'perfor', 'sex', 'obstruct', 'adhere', 'surg', 'extent']
full_data = colon[covariates + ['time', 'status']]


# Define function to get the kernel

def helper_function(covariate):
    if covariate in ['age', 'extent']:
        kernel = 'gau'
    else:
        kernel = 'bin'
    return kernel

def get_kernels_covariates(covariates):
    if type(covariates) == str:
        kernel = helper_function(covariates)
    else:
        kernel = []
        for covariate in covariates:
            kernel.append(helper_function(covariate))
    return kernel



covariates_list = [
    'age',  # 1
    ['age', 'perfor'],  # 2
    ['age', 'sex'],  # 3
    ['obstruct', 'perfor'],  # 4
    ['age', 'obstruct', 'perfor'],  # 5
    ['age', 'obstruct', 'sex'],  # 6
    ['age', 'perfor', 'adhere'],  # 7
    ['age', 'perfor', 'sex'],  # 8
    ['age', 'adhere', 'sex'],  # 9
    ['age', 'sex', 'surg'],  # 10
    ['obstruct', 'sex', 'surg'],
    ['adhere', 'sex', 'extent'],
    ['adhere', 'sex', 'surg'],
    ['sex', 'extent', 'surg']
]

for covariates in covariates_list: # the two kernels for this scenario
    print('covariates:', covariates)
    p_value_dict = {
        'cph': 0,
        'gaucon': 0,
        'gaugau': 0
    }

    data = full_data
    n = data.shape[0]
    x = np.array(data[covariates]).reshape(n, -1)
    z = np.array(data.time)
    d = np.array(data.status)

    kx = get_kernels_covariates(covariates)

    # Gaucon
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
        x=x, z=z, d=d, kernels_x=kx, kernel_z='con', num_bootstrap_statistics=B)
    p_value_dict['gaucon'] += p

    # Gaugau
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
        x=x, z=z, d=d, kernels_x=kx, kernel_z='gau', num_bootstrap_statistics=B)
    p_value_dict['gaugau'] += p

    # CPH
    try:
        p = CPH_test(x=x, z=z, d=d)
    except:
        p = p_value_dict['cph']
    p_value_dict['cph'] += p

    for key, val in p_value_dict.items():
        p_value_dict[key] = p_value_dict[key]

    print(p_value_dict)
