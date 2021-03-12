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
from tqdm import tqdm
import h5py

np.random.seed(1)

# Load data
covariates = ['bfb']

biofeedback = pd.read_csv('../../data/biofeedback.txt', sep='\t', lineterminator='\n')
biofeedback.to_csv('../../data/biofeedback_csv.csv')

biofeedback.to_csv('../../data/biofeedback_csv.csv')

full_data = biofeedback
B = 10000
sample_size = 33
num_repetitions = 1

print('mean', biofeedback.success.mean())

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

kernels = [
    # ['linfis', 'con'],
    # ['lin', 'con'],
    [['bin','gau'], 'con'],
    [['bin','gau'], 'gau'],
    # ['lin', 'gau'],
]

p_value_dict = {}
p_value_dict['cph'] = 0
kernel_names = ['gaucon', 'gaugau']
for name in kernel_names:
    p_value_dict[name] = 0

for repetition in range(num_repetitions):
    data = full_data.sample(sample_size)
    x = np.array(data[covariates])
    z = np.array(data.thdur)
    d = np.array(data.success)
    for kernel, kernel_name in zip(kernels, kernel_names):
        kx, kz = kernel
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
            x=x, z=z, d=d, kernels_x=kx, kernel_z=kz, num_bootstrap_statistics=B, fast_computation=False)
        p_value_dict[kernel_name] += p
        print(p)
    try:
        p = CPH_test(x=x, z=z, d=d)
    except:
        print('CPH test failed')
        p = p_value_dict['cph_test']
        print('except')
    p_value_dict['cph'] += p

for key, val in p_value_dict.items():
    p_value_dict[key] = val / num_repetitions

print('p value dict', p_value_dict)

# to_pickle_dict = {'p_values': p_value_dict, 'parameters': {
#     'sample_size': sample_size,
#     'kernels': kernels,
#     'B': B,
#     'num_repetitions': num_repetitions,
#     'covariates': covariates
# }}
