 # Kernel Logrank and OptHsic

### Contents

This repository contains python code for methods to test independence between a right-censored survival time T (censored by a censoring time C) and a covariate X, a random vector in any dimension. The methods are the Kernel Log rank test (1) and OptHsic (2). For the methods see

(1) A kernel log-rank test of independence for right-censored data/ Tamara Fernandez, Arthur Gretton, David Rindt, Dino Sejdinovic. (https://arxiv.org/abs/1912.03784).

(2) Nonparametric Independence Testing for Right-Censored Data using Optimal Transport
David Rindt, Dino Sejdinovic, David Steinsaltz. (https://arxiv.org/abs/1906.03866)

The code for 1) is found in kernel_logrank.tests.wild_bootstrap_test_logrank_covariates.py
The code for 2) is found in kernel_logrank.tests.opt_hsic.py.
A use example is in the file example.py. 

### Dependencies

This repository uses the packages kerpy (https://github.com/oxmlcs/kerpy), pot (https://pot.readthedocs.io/en/stable/), lifelines (https://lifelines.readthedocs.io/en/latest/) and scipy, numpy, pandas, pickle, multiprocessing, matplotlib, dcor (https://pypi.org/project/dcor/). 

### Questions and Issues

Any questions and issues can be sent to David Rindt, (firstname.lastname@stats.ox.ac.uk).
