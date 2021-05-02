import numpy as np
import pandas as pd
from kernel_logrank.tests import wild_bootstrap_LR
from kernel_logrank.tests.cph_test import cph_test

# Set some numbers
B = 1000
seed = 1

# Load data
colon = pd.read_csv('../data/colon')
colon = colon[colon.etype == 2]
covariates = ['age', 'perfor', 'sex', 'obstruct', 'adhere', 'surg', 'extent']
full_data = colon[covariates + ['time', 'status']]

print('percentage observed', full_data.status.mean())


# Define a function to get the kernel
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
    'age',
    ['age', 'perfor', 'adhere'],
]

for covariates in covariates_list:  # the two kernels for this scenario
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
    print('kernels:', kx)

    # Gaucon
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=x, z=z, d=d, kernels_x=kx, kernel_z='con',
                                                                    seed=seed, num_bootstrap_statistics=B)
    p_value_dict['gaucon'] += p

    # Gaugau
    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(X=x, z=z, d=d, kernels_x=kx, kernel_z='gau',
                                                                    seed=seed, num_bootstrap_statistics=B)
    p_value_dict['gaugau'] += p

    # CPH
    p = cph_test(X=x, z=z, d=d)
    p_value_dict['cph'] += p

    for key, val in p_value_dict.items():
        p_value_dict[key] = p_value_dict[key]

    print(p_value_dict)
