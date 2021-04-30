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

np.random.seed(1)

# Load data

loan_data = pd.read_csv('../../kernel_logrank/data/loan_data')
IsBorrowerHomeowner_map = {True: 0,
                           False: 1}
loan_data.IsBorrowerHomeowner = loan_data.IsBorrowerHomeowner.map(IsBorrowerHomeowner_map)
print(loan_data.shape)
full_data = loan_data
covariates = ['LoanOriginalAmount2', 'IsBorrowerHomeowner']
to_save_data = full_data[['status', 'time', 'LoanOriginalAmount2', 'IsBorrowerHomeowner' ]]
to_save_data.to_csv('selected_loan_data.csv')
# Do experiment

print('shape', full_data.shape)
print('mean', full_data.status.mean())

sample_size = 1000
B = 5000
num_repetitions = 1000
kernels = [
    # ['linfis', 'con'],
    # ['lin', 'con'],
    [['gau', 'bin'], 'con'],
    [['gau', 'bin'], 'gau'],
    # ['lin', 'gau'],
]
p_value_dict = {}
p_value_dict['cph_test'] = 0
for kx, kz in kernels:
    p_value_dict[kx[0] + kz] = 0
#
# for repetition in tqdm(range(num_repetitions)):
#     data = pd.DataFrame(full_data.sample(sample_size))
#     x = np.array(data[covariates])
#     z = np.array(data.time)
#     d = np.array(data.status)
#     for kernel in kernels:
#         kx, kz = kernel
#         v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
#             x=x, z=z, d=d, kernels_x=kx, kernel_z=kz, num_bootstrap_statistics=B, fast_computation=False)
#         p_value_dict[kx[0] + kz] += p
#     try:
#         p = CPH_test(x=x, z=z, d=d)
#     except:
#         print('CPH test failed')
#         p = p_value_dict['cph_test'] / (num_repetitions + 1)
#         print('except')
#     p_value_dict['cph_test'] += p
#
# for key, val in p_value_dict.items():
#     p_value_dict[key] = p_value_dict[key] / num_repetitions
#
# print('p value dict', p_value_dict)
#
# to_pickle_dict = {'p_values': p_value_dict, 'parameters': {
#     'sample_size': sample_size,
#     'kernels': kernels,
#     'B': B,
#     'num_repetitions': num_repetitions,
#     'covariates': covariates
# }}

# if num_repetitions >9 and B >9:
#     with open('loan_data_result.pickle', 'wb') as f:
#         pickle.dump(to_pickle_dict, f)
#
#
#     with open('loan_data_result.pickle', 'rb') as f:
#         dic = pickle.load(f)
#         print('loaded dic', dic)