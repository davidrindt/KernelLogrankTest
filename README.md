 # Kernel Logrank and OptHsic

### Introduction

This repository contains python code for two methods to test independence between a right-censored survival time T (censored by a censoring time C) and a covariate X, a random vector in any dimension. The methods are the Kernel Log rank test (1) and OptHsic (2). For the methods see

(1) A kernel log-rank test of independence for right-censored data/ Tamara Fernandez, Arthur Gretton, David Rindt, Dino Sejdinovic. [url](https://arxiv.org/abs/1912.03784).

(2) Nonparametric Independence Testing for Right-Censored Data using Optimal Transport
David Rindt, Dino Sejdinovic, David Steinsaltz. [url](https://arxiv.org/abs/1906.03866)

### Contents

1. wild_bootstrap_LR.py contains the kernel log rank test of (1), that uses wild bootstrap to reject or accept the null hypothesis. 
2. opthsic.py contains the opthsic test of (2).
3. FisherInformation computes the Fisher Information matrix and it's inverse (used for the Fisher kernel of (1)).
4. CPH_test.py performs the standard Cox Proportional Hazards Likelihood ratio test.
5. examples.py simulates an example dataset and runs the different tests on this simulated dataset.
6. all the 'exp_' files, exp_dim_range.py, exp_larger_n_values_range.py, exp_n_values_range.py, exp_parameter_range.py contain experiments of (1). These can be easily replicated. Running the code will ask for a scenario, method and choices of kernels. Note one may want to change the number of repeitions to calculate the power, depending on the time available.

### Dependencies

This repository uses the packages kerpy [url](https://github.com/oxmlcs/kerpy), pot [url](https://pot.readthedocs.io/en/stable/), lifelines [url](https://lifelines.readthedocs.io/en/latest/) and scipy, numpy, pandas, pickle, multiprocessing, matplotlib, dcor [url](https://pypi.org/project/dcor/). 

### Questions and Issues

Any questions and issues can be sent to David Rindt, (firstname.lastname@stats.ox.ac.uk).
