#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:19:48 2021

@author: rindt
"""

import sys
sys.path.append('../../tests')
sys.path.append('../../utils')
import numpy as np
from wild_bootstrap_LR import wild_bootstrap_test_logrank_covariates
from lifelines.statistics import logrank_test
import pandas as pd


gastric = pd.read_csv('../../data/gastric.csv')
print(gastric)

# V1 is the time, V2 is the censor indicator, V3 is the group
z = gastric.V1.values
d = gastric.V2.values
x = gastric.V3.values
x = x[:, None]
print(wild_bootstrap_test_logrank_covariates(x, z, d, kernel_x='euc', kernel_z='gau', seed=1, num_bootstrap_statistics = 1000000, fast_computation = True))



durations_A = z[np.where(x.flatten()==0)]
durations_B = z[np.where(x.flatten()==1)]
event_observed_A = d[np.where(x.flatten()==0)]
event_observed_B = d[np.where(x.flatten()==1)]
print(durations_A, durations_B)
print(logrank_test(durations_A, durations_B, event_observed_A, event_observed_B))