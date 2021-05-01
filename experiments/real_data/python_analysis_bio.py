#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:54:29 2021

@author: rindt
"""

import sys

sys.path.append('../../kernel_logrank/utils')
sys.path.append('../data')
sys.path.append('../../kernel_logrank/tests')
from CPH_test import CPH_test
import numpy as np
import pandas as pd
import wild_bootstrap_LR  as wild_bootstrap_LR

# Load data
seed=1

covariates = ['bfb']

biofeedback = pd.read_csv('../data/biofeedback.txt', sep='\t', lineterminator='\n')
biofeedback.to_csv('../../data/biofeedback_csv.csv')

biofeedback.to_csv('../../data/biofeedback_csv.csv')
print(biofeedback.head())
full_data = biofeedback
B = 10000
num_repetitions = 1

print('mean', biofeedback.success.mean())

covariates_list = [
    'bfb',
    'theal',
    ['bfb', 'theal'],
    ['bfb', 'log2heal']
]
def helper_function(covariate):
    if covariate in ['bfb']:
        kernel = 'bin'
    else:
        kernel = 'gau'
    return kernel

def get_kernels_covariates(covariates):
    if type(covariates) == str:
        kernel = helper_function(covariates)
    else:
        kernel = []
        for covariate in covariates:
            kernel.append(helper_function(covariate))
    return kernel



for covariates in covariates_list:
    print('covariates:', covariates)
    p_value_dict = {
        'cph': 0,
        'gaucon': 0,
        'gaugau': 0
    }

    data = full_data
    n = data.shape[0]
    x = np.array(data[covariates]).reshape(n, -1)
    z = np.array(data.thdur)
    d = np.array(data.success)

    kx = get_kernels_covariates(covariates)
    print('kernels:', kx)

    # Gaucon
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
        x=x, z=z, d=d, kernels_x=kx, kernel_z='con', num_bootstrap_statistics=B, seed=seed)
    p_value_dict['gaucon'] += p

    # Gaugau
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
        x=x, z=z, d=d, kernels_x=kx, kernel_z='gau', num_bootstrap_statistics=B, seed=seed)
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
# to_pickle_dict = {'p_values': p_value_dict, 'parameters': {
#     'sample_size': sample_size,
#     'kernels': kernels,
#     'B': B,
#     'num_repetitions': num_repetitions,
#     'covariates': covariates
# }}
