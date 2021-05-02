import numpy as np
from multiprocessing import Pool
from kernel_logrank.tests.cph_test import cph_test
from kernel_logrank.tests.wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from kernel_logrank.tests.opt_hsic import opt_hsic
import sys

np.random.seed(1)
num_repetitions = 1000

scenario = int(input('enter number\n'))
method = input('enter method, choose from cph lr opt\n')
if method == 'cph':
    kernel_x = ''
    kernel_z = ''
elif method == 'lr':
    method = 'lr'
    kernel_x = input('enter kernel_x, choose from lin linfis gaufis gau pol\n')
    kernel_z = input('enter kernel_z choose from con gau\n')
elif method == 'opt':
    kernel_x = ''
    kernel_z = ''
else:
    sys.exit('choose a valid method')

M = np.random.normal(size=22 ** 2).reshape(22, 22)
psd = np.matmul(M, np.transpose(M))


def rejection_rate(a):
    n = 100
    num_observed = 0
    seed = a[0]
    theta = a[1]
    num_rejections = 0
    local_state = np.random.RandomState(seed)
    seeds_for_test = local_state.choice(1000000, size=num_repetitions, replace=False)
    for repetition in range(num_repetitions):
        if repetition % 100 == 0:
            print('n is', n, 'repetition', repetition)
        # GENERATE DATA

        # True rejections
        if scenario == 1:
            x = local_state.normal(loc=0, scale=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp(x[v] * theta))
            c = local_state.exponential(scale=1.5, size=n)
            x = x[:, np.newaxis]

        elif scenario == 2:
            x = local_state.normal(loc=0, scale=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp((theta * (x[v] ** 2))))
            c = local_state.exponential(scale=1.5, size=n)
            x = x[:, np.newaxis]

        else:
            x = 0
            x = x[:, np.newaxis]
            t = 0
            c = 0
            z = 0
            n = 0
            print('ERROR')
        d = np.int64(c > t)
        z = np.minimum(t, c)

        # do the test
        if method == 'cph':
            num_rejections += 1 if cph_test(x=x, z=z, d=d) <= 0.05 else 0
        elif method == 'lr':
            num_rejections += 1 if wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d,
                                                                          seed=seeds_for_test[repetition],
                                                                          kernel_x=kernel_x,
                                                                          kernel_z=kernel_z) <= 0.05 else 0
        else:
            num_rejections += 1 if opt_hsic(x=x, z=z, d=d, seed=seeds_for_test[repetition]) <= 0.05 else 0
    print('percentage observed', num_observed / (n * num_repetitions))
    return num_rejections / num_repetitions


filename = 'pickles/parameters_exp_cens_' + method + '_' + str(scenario) + '_' + kernel_x + kernel_z + '.pickle'
print(filename)

parameters = np.arange(-.4, .5, step=.1)
seeds = np.random.choice(10000, replace=False, size=len(parameters))
inputs = [[seeds[i], parameters[i]] for i in range(len(parameters))]
print(inputs)
p = Pool()
rejection_rate_vector = p.map(rejection_rate, inputs)
p.close()
p.join()
print('result', rejection_rate_vector)

output_dict = {'parameters': parameters, 'rejection_rate': rejection_rate_vector}
print(output_dict)
