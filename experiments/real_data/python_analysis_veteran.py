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
import wild_bootstrap_LR  as wild_bootstrap_LR

np.random.seed(1)

# Load data

covariates = ['treat', 'age', 'Karn', 'diag.time', 'cell', 'prior']
covariates = ['treat', 'prior', 'cell', 'age', 'diag.time']
covariates = ['treat', 'prior', 'cell']
covariates = ['treat', 'prior', 'cell', 'diag.time', 'age']
covariates_cph = ['treat', 'prior', 'squamos', 'small', 'adeno', 'diag.time', 'age']
covariates = ['treat', 'diag.time' ]

covariates_cph = covariates
print(len(covariates))

veteran_data = pd.read_csv('../../kernel_logrank/data/VA.csv')
veteran_data = pd.get_dummies(veteran_data)
cell_types = ['squamos', 'small', 'adeno', 'large']
cell_code = [1, 2, 3, 4]
for cell_type, i in zip(cell_types, cell_code):
    veteran_data[cell_type] = 0
    veteran_data.loc[veteran_data.cell == i, cell_type] = 1

veteran_data.drop(['Karn'], axis=1, inplace=True)

treat_map = {1:0, 2:1}
prior_map = {0:0, 10:1}

veteran_data.treat = veteran_data.treat.map(treat_map)
veteran_data.prior = veteran_data.prior.map(prior_map)
veteran_data.drop(['Unnamed: 0'], axis=1, inplace=True)
full_data = veteran_data
print(veteran_data.columns.values)
print(veteran_data)
print(veteran_data.nunique())
print(veteran_data.status.unique())
print(veteran_data.treat.unique())
print(veteran_data.prior.unique())
print(veteran_data.describe())



# IsBorrowerHomeowner_map = {True: 0,
#                            False: 1}
# loan_data.IsBorrowerHomeowner = loan_data.IsBorrowerHomeowner.map(IsBorrowerHomeowner_map)
# print(loan_data.shape)
# full_data = loan_data
# covariates = ['LoanOriginalAmount2', 'IsBorrowerHomeowner']
# to_save_data = full_data[['status', 'time', 'LoanOriginalAmount2', 'IsBorrowerHomeowner' ]]
# to_save_data.to_csv('selected_loan_data.csv')


B = 10000
sample_size = 137
num_repetitions = 1

# kernels = [
#     # ['linfis', 'con'],
#     # ['lin', 'con'],
#     [['gau', 'gau', 'gau', 'gau', 'bin', 'bin', 'bin', 'bin', 'gau'], 'con'],
#     [['gau', 'gau', 'gau', 'gau', 'bin', 'bin', 'bin', 'bin', 'gau'], 'gau'],
#     # ['lin', 'gau'],
# ]

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
    [['bin', 'bin', 'bin', 'gau', 'gau'], 'con'],
    [['bin', 'bin', 'bin', 'gau', 'gau'], 'gau'],
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
    z = np.array(data.stime)
    d = np.array(data.status)
    for kernel, kernel_name in zip(kernels, kernel_names):
        kx, kz = kernel
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
            x=x, z=z, d=d, kernels_x=kx, kernel_z=kz, num_bootstrap_statistics=B, fast_computation=False)
        p_value_dict[kernel_name] += p
        print(p)
    x = np.array(data[covariates_cph])
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
