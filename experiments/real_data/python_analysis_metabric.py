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
from CPH_test import CPH_test
import numpy as np
import pandas as pd
import wild_bootstrap_LR
import h5py
from tqdm import tqdm

np.random.seed(1)

covariate_columns={i:'x' + str(i) for i in range(9)}
covariates = covariate_columns.values()
with h5py.File('../../data/metabric_IHC4_clinical_train_test.h5', 'r') as f:
    train_data = f['train']
    x_full = pd.DataFrame(train_data['x']).rename(columns=covariate_columns)
    z_full = pd.DataFrame(train_data['t']).rename(columns={0:'z'})
    d_full = pd.DataFrame(train_data['e']).rename(columns={0:'d'})
    full_data = pd.DataFrame(pd.concat([x_full, z_full, d_full], axis=1))




sample_size = 300
B = 500
num_repetitions = 20
kernels = [
    # ['linfis', 'con'],
    # ['lin', 'con'],
    [['gau', 'gau', 'gau', 'gau', 'bin', 'bin', 'bin', 'bin', 'gau'], 'con'],
    [['gau', 'gau', 'gau', 'gau', 'bin', 'bin', 'bin', 'bin', 'gau'], 'gau'],
    # ['lin', 'gau'],
]

kernels = [
    # ['linfis', 'con'],
    # ['lin', 'con'],
    ['gau', 'con'],
    ['gau', 'gau'],
    # ['lin', 'gau'],
]

p_value_dict = {}
p_value_dict['cph_test'] = 0
for kx, kz in kernels:
    p_value_dict[kx[0] + kz] = 0

for repetition in tqdm(range(num_repetitions)):
    data = pd.DataFrame(full_data.sample(sample_size))
    x = np.array(data[covariates])
    z = np.array(data.z)
    d = np.array(data.d)
    for kernel in kernels:
        kx, kz = kernel
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
            x=x, z=z, d=d, kernels_x=kx, kernel_z=kz, num_bootstrap_statistics=B, fast_computation=False)
        p_value_dict[kx[0] + kz] += p
    try:
        p = CPH_test(x=x, z=z, d=d)
    except:
        print('CPH test failed')
        p = p_value_dict['cph_test'] / (num_repetitions + 1)
        print('except')
    p_value_dict['cph_test'] += p

for key, val in p_value_dict.items():
    p_value_dict[key] = p_value_dict[key] / num_repetitions

print('p value dict', p_value_dict)

to_pickle_dict = {'p_values': p_value_dict, 'parameters': {
    'sample_size': sample_size,
    'kernels': kernels,
    'B': B,
    'num_repetitions': num_repetitions,
    'covariates': covariates
}}
