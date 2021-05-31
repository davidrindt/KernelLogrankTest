import numpy as np
from kernel_logrank.tests.cph_test import cph_test
from kernel_logrank.tests.wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from kernel_logrank.tests.opt_hsic import opt_hsic
from kernel_logrank.utils.generate_synthetic_data import generate_synthetic_data

X, z, d = generate_synthetic_data()

stat0, p0 = wild_bootstrap_test_logrank_covariates(X, z, d, 'linfis', 'con')
stat1, p1 = wild_bootstrap_test_logrank_covariates(X, z, d, 'lin', 'con')
stat2, p2 = wild_bootstrap_test_logrank_covariates(X, z, d, 'gau', 'con')
stat3, p3 = wild_bootstrap_test_logrank_covariates(X, z, d, 'gau', 'gau')
p4 = cph_test(X, z, d)

print(f'pvalue 1 {p1}')
print(f'pvalue 2 {p2}')
print(f'pvalue 3 {p3}')
print(f'pvalue 4 {p4}')
