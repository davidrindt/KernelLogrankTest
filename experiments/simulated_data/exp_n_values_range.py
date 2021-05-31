import numpy as np
from multiprocessing import Pool
from kernel_logrank.tests.cph_test import cph_test
from kernel_logrank.tests.wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from kernel_logrank.tests.opt_hsic import opt_hsic
import sys
from scipy.linalg import expm

np.random.seed(1)

# Define an orthonormal matrix used in scenario 31
a = np.random.normal(0, 1, size=(10, 10))
r = (a - np.transpose(a)) / 2
orth = expm(r)

scenario = int(input('enter number\n'))

num_repetitions = 10
if scenario >= 20:
    num_repetitions = 10
if scenario >= 30:
    num_repetitions = 10

method = input('enter method, choose from cph lr opt\n')
if method == 'cph':
    kernel_x = ''
    kernel_z = ''
elif method == 'lr':
    method = 'lr'
    kernel_x = input('enter kernel_x, choose from lin linfis gau pol\n')
    kernel_z = input('enter kernel_z choose from con gau\n')
elif method == 'opt':
    kernel_x = ''
    kernel_z = ''
else:
    sys.exit('choose a valid method')

M = np.random.normal(size=10 ** 2).reshape(10, 10)
psd = np.matmul(M, np.transpose(M))


def rejection_rate(a):
    num_observed = 0
    seed = a[0]
    n = a[1]
    num_rejections = 0
    local_state = np.random.RandomState(seed)
    seeds_for_test = local_state.choice(1000000, size=num_repetitions, replace=False)
    for repetition in range(num_repetitions):
        if repetition % 100 == 0:
            print('n is', n, 'repetition', repetition)
        # GENERATE DATA

        # True rejections
        if scenario == 1:
            x = local_state.uniform(low=-1, high=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp(x[v] / 3))
            c = local_state.exponential(scale=1.5, size=n)
            x = x[:, np.newaxis]

        elif scenario == 2:
            x = local_state.uniform(low=-1, high=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp(x[v] ** 2))
            c = local_state.exponential(scale=2.25, size=n)
            x = x[:, np.newaxis]

        elif scenario == 3:
            x = local_state.uniform(low=-1, high=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.weibull(x[v] * 1.75 + 3.25)
            c = local_state.exponential(1.75, size=n)
            x = x[:, np.newaxis]

        elif scenario == 4:
            x = local_state.uniform(low=-1, high=1, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.normal(loc=100 - x[v] * 2.25, scale=x[v] * 4.5 + 5.5)
            c = 82 + local_state.exponential(35, size=n)
            x = x[:, np.newaxis]

        # Scenario 5 will be generated in the file: exp_larger_n, this is the complicated relationship

        elif scenario == 6:
            dim = 10

            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=psd, size=n)
            t = np.zeros(n)
            rowsum = np.sum(x, axis=1)
            for v in range(n):
                t[v] = local_state.exponential(np.exp((rowsum[v] / 60)))
            c = local_state.exponential(1.5, size=n)

        elif scenario == 7:
            dim = 10

            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=psd, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp((x[v, 0] / 8)))
            c = local_state.exponential(1.5, size=n)

        elif scenario == 8:
            dim = 10

            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=psd, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp((((x[v, 0]) ** 2) / 8)))
            c = local_state.exponential(3, size=n)

        # False rejections
        elif scenario == 20:
            x = local_state.uniform(low=-1, high=1, size=n)
            c = np.zeros(n)
            for v in range(n):
                c[v] = local_state.exponential(np.exp(x[v]))
            t = local_state.exponential(scale=2 / 3, size=n)
            x = x[:, np.newaxis]

        elif scenario == 21:
            x = local_state.uniform(low=-1, high=1, size=n)
            c = np.zeros(n)
            for v in range(n):
                c[v] = local_state.exponential(np.exp(3 * x[v] ** 2))
            t = local_state.exponential(scale=1.6, size=n)
            x = x[:, np.newaxis]

        elif scenario == 22:
            x = local_state.uniform(low=-1, high=1, size=n)
            c = np.zeros(n)
            for v in range(n):
                c[v] = local_state.weibull(x[v] * 1.75 + 3.25)
            t = local_state.exponential(.9, size=n)
            x = x[:, np.newaxis]

        elif scenario == 23:
            x = local_state.uniform(low=-1, high=1, size=n)
            c = 1 + x
            t = local_state.exponential(scale=.9, size=n)
            x = x[:, np.newaxis]

        elif scenario == 24:
            dim = 10
            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=psd[0:dim, 0:dim], size=n)
            c = np.zeros(n)
            rowsum = np.sum(x, axis=1)
            for v in range(n):
                c[v] = local_state.exponential(np.exp((rowsum[v] / 8)))
            t = local_state.exponential(.6, size=n)

        elif scenario == 25:
            dim = 10

            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=psd[0:dim, 0:dim], size=n)
            c = np.zeros(n)
            for v in range(n):
                c[v] = local_state.exponential(np.exp((x[v, 0] / 8)))
            t = local_state.exponential(.6, size=n)



        elif scenario == 30:
            dim = 10
            covmatrix = np.identity(10)
            covmatrix[0, 0] = 1 / 10

            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=covmatrix, size=n)
            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp(x[v, 0]))
            c = local_state.exponential(1.5, size=n)
        elif scenario == 31:
            dim = 10
            covmatrix = np.identity(10)
            covmatrix[0, 0] = 1 / 10
            x = local_state.multivariate_normal(mean=np.zeros(dim), cov=covmatrix, size=n)

            tilde_x = np.matmul(x, np.transpose(orth))

            t = np.zeros(n)
            for v in range(n):
                t[v] = local_state.exponential(np.exp(x[v, 0]))
            c = local_state.exponential(1.5, size=n)
            x = tilde_x
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
        num_observed += np.sum(d)

        # do the test
        if method == 'cph':
            num_rejections += 1 if cph_test(X=x, z=z, d=d) <= 0.05 else 0
        elif method == 'lr':
            num_rejections += 1 if wild_bootstrap_test_logrank_covariates(X=x, z=z, d=d,
                                                                          seed=seeds_for_test[repetition],
                                                                          kernel_x=kernel_x,
                                                                          kernel_z=kernel_z) <= 0.05 else 0
        else:
            num_rejections += 1 if opt_hsic(X=x, z=z, d=d, seed=seeds_for_test[repetition]) <= 0.05 else 0
    print('percentage observed', num_observed / (n * num_repetitions))
    return num_rejections / num_repetitions


filename = 'pickles/exp_cens_' + method + '_' + str(scenario) + '_' + kernel_x + kernel_z + '.pickle'
print(filename)
seeds = np.random.choice(10000, replace=False, size=7)
n_values = [50, 100, 150, 200, 250, 300, 350]
inputs = [[seeds[i], n_values[i]] for i in range(7)]
print(inputs)
p = Pool()
rejection_rate_vector = p.map(rejection_rate, inputs)
p.close()
p.join()
print('result', rejection_rate_vector)

output_dict = {'n_values': n_values, 'rejection_rate': rejection_rate_vector}
print(output_dict)
