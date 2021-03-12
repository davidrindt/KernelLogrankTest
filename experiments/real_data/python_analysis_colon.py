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

colon = pd.read_csv('../../data/colon')
colon = colon[colon.etype == 2]
colon = pd.get_dummies(colon)
full_data = colon

# print(colon)
# print(colon.age, colon.sex, colon.perfor, colon.adhere, colon.surg, colon.perfor, colon.obstruct, colon.obstruct)
# print(colon.perfor.sum())  # perfor has very few ones (way less than 1 percent)

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
    ['age', 'sex', 'surg']  # 10
]

kernels_list = [
    [['gau', 'con'], ['gau', 'gau']],  # 1
    [[['gau', 'bin'], 'con'], [['gau', 'bin'], 'gau']],  # 2
    [[['gau', 'bin'], 'con'], [['gau', 'bin'], 'gau']],  # 3
    [[['bin', 'bin'], 'con'], [['bin', 'bin'], 'gau']],  # 4
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 5
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 6
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 7
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 8
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 9
    [[['gau', 'bin', 'bin'], 'con'], [['gau', 'bin', 'bin'], 'gau']],  # 10
]

kernel_parameters_list = [
    [[3., None], [3., 0.5]],  # 1
    [[[3., None], None], [[3., None], 0.5]],  # 2
    [[[3., None], None], [[3., None], 0.5]],  # 3
    [[[None, None], None], [[None, None], 0.5]],  # 4
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 5
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 6
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 7
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 8
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 9
    [[[3., None, None], None], [[3., None, None], 0.5]],  # 10
]

kernel_parameters_list = [
    [[None, None], [None, None]],  # 1
    [[None, None], [None, None]],  # 2
    [[None, None], [None, None]],  # 3
    [[None, None], [None, None]],  # 4
    [[None, None], [None, None]],  # 5
    [[None, None], [None, None]],  # 6
    [[None, None], [None, None]],  # 7
    [[None, None], [None, None]],  # 8
    [[None, None], [None, None]],  # 9
    [[None, None], [None, None]],  # 10
]



kernel_names = [
    'gaucon', 'gaugau'
]

B = 1000

for kernels, kernel_parameters, covariates in zip(kernels_list, kernel_parameters_list, covariates_list): # the two kernels for this scenario
    print('covariates:', covariates)
    p_value_dict = {
        'cph': 0,
        'gaucon': 0,
        'gaugau': 0
    }
    # data = pd.DataFrame(full_data.sample(sample_size))
    data = full_data
    n = data.shape[0]
    x = np.array(data[covariates]).reshape(n, -1)
    z = np.array(data.time)
    d = np.array(data.status)
    for kernel_name, kernel_parameter, kernel in zip(kernel_names, kernel_parameters, kernels): # select one of the kernels
        kx, kz = kernel # split into x and z kernel
        print(get_kernel_matrix.get_total_kernel_matrix(x, kx, kernel_parameters=None, d=None))
        # print('kernel x', kx, 'kernel z', kz)
        x_parameters, z_parameters = kernel_parameter
        # print(kernel, kernel_parameter)
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(
            x=x, z=z, d=d, kernels_x=kx, kernel_z=kz, kernel_parameters_x=x_parameters, kernel_parameters_z=z_parameters, num_bootstrap_statistics=B)
        p_value_dict[kernel_name] += p
    try:
        p = CPH_test(x=x, z=z, d=d)
    except:
        p = p_value_dict['cph']
    p_value_dict['cph'] += p

    for key, val in p_value_dict.items():
        p_value_dict[key] = p_value_dict[key]

    print(p_value_dict)
