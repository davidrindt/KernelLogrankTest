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
import numpy as np
import pandas as pd

np.random.seed(1)

# Load data

veteran_data = pd.read_csv('../../data/VA')
print(veteran_data.head())
# IsBorrowerHomeowner_map = {True: 0,
#                            False: 1}
# loan_data.IsBorrowerHomeowner = loan_data.IsBorrowerHomeowner.map(IsBorrowerHomeowner_map)
# print(loan_data.shape)
# full_data = loan_data
# covariates = ['LoanOriginalAmount2', 'IsBorrowerHomeowner']
# to_save_data = full_data[['status', 'time', 'LoanOriginalAmount2', 'IsBorrowerHomeowner' ]]
# to_save_data.to_csv('selected_loan_data.csv')