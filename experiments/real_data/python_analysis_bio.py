import numpy as np
import pandas as pd
from kernel_logrank.tests import wild_bootstrap_LR
from kernel_logrank.tests import cph_test

# Load data amd set some numbers
seed = 1
full_data = pd.read_csv('../data/biofeedback.txt', sep='\t', lineterminator='\n')
B = 10000
num_repetitions = 1

print('mean', full_data.success.mean())

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
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=x, z=z, d=d, kernels_x=kx, kernel_z='con',
                                                                    seed=seed, num_bootstrap_statistics=B)
    p_value_dict['gaucon'] = p

    # Gaugau
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=x, z=z, d=d, kernels_x=kx, kernel_z='gau',
                                                                    seed=seed, num_bootstrap_statistics=B)
    p_value_dict['gaugau'] = p

    # CPH
    p = cph_test.cph_test(X=x, z=z, d=d)
    p_value_dict['cph'] = p

    for key, val in p_value_dict.items():
        p_value_dict[key] = p_value_dict[key]

    print(p_value_dict)
