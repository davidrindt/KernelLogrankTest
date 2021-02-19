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
from survival_scatter_plot import survival_scatter_plot
import wild_bootstrap_LR 
import pickle
diabetic = pd.read_csv('../../data/diabetic.csv')
diabetic.laser = pd.to_numeric(diabetic.laser, errors='ignore')

for i, las in enumerate(diabetic.laser):
    print(i, las)
    if las == 'argon':
        diabetic.laser.values[i] = np.float(1)
    else:
        diabetic.laser.values[i] = np.float(0)

for i, ey in enumerate(diabetic.eye):
    print(i, ey)
    if ey == 'left':
        diabetic.eye.values[i] = np.float(1)
    else:
        diabetic.eye.values[i] = np.float(0)

print(diabetic)
diabetic_no_trt = diabetic[diabetic.trt == 0]
diabetic_trt = diabetic[diabetic.trt == 1]

#
## Treatment group. Dependence risk with time
#print('trt. Risk vs time')
#
#grp = diabetic_trt
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#survival_scatter_plot(x=x, z=z, d=d)
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
## Treatment group and left eye. Dependence risk with time
#
#print('trt, left. Risk vs time')
#
#grp = diabetic_trt[diabetic_trt.eye == 'left']
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
## Treatment group and right eye. Dependence risk with time
#
#print('trt, right. Risk vs time')
#grp = diabetic_trt[diabetic_trt.eye == 'right']
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
## No treatment group. Dependence risk with time
#print('No trt. Risk vs time')
#
#grp = diabetic_no_trt
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
## No treatment group. Dependence risk with time
#print('No trt left. Risk vs time')
#grp = diabetic_no_trt[diabetic_no_trt.eye == 'left']
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
## No treatment group. Dependence risk with time
#print('No trt right. Risk vs time')
#grp = diabetic_no_trt[diabetic_no_trt.eye == 'right']
#x, z, d = grp.risk.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
## No treatment group. Dependence age with time
#print('No trt. Age vs time')
#
#grp = diabetic_no_trt
#x, z, d = grp.age.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
## Treatment group. Dependence age with time
#print('Trt. Age vs time')
#
#grp = diabetic_trt
#x, z, d = grp.age.values[:,None], grp.time.values, grp.status.values
#
#
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='lin', kernel_z='con')
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='con')
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=x, z=z, d=d, kernel_x='gau', kernel_z='gau')
#print('gua, gau', p, v)
#
#
#
## All data
#print('\n \n FULL DATA \n \n')
#print(diabetic.head())
#s_diabetic = diabetic[['time', 'status', 'age', 'trt', 'risk']]
#
#X = np.array(s_diabetic[['age', 'trt', 'risk']])
#z = s_diabetic.time.values
#d = s_diabetic.status.values
#print(s_diabetic)
#print(X)
#
#B = int(10e2)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='lin', kernel_z='con', num_bootstrap_statistics=B)
#print('lin, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='con', num_bootstrap_statistics=B)
#print('gau, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='gau', num_bootstrap_statistics=B)
#print('gua, gau', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='con', num_bootstrap_statistics=B)
#print('linfis, con', p, v)
#v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='gau', num_bootstrap_statistics=B)
#print('linfis, gau', p, v)
#
#print('CPH test', CPH_test(X, z, d))


# Subsample rows

print('\n \n SUBSAMPLED DATA \n \n')


selected_variables_diabetic = diabetic_trt[['time', 'status', 'age', 'trt', 'risk']]
selected_variables_diabetic.head()

B = int(1000)
num_repetitions = 1000

variables_list = [
        ['laser', 'age', 'eye', 'trt', 'risk'],
        ['laser', 'age', 'eye', 'trt', 'risk'],
        ['trt'],
        ['trt'],
        ['age','risk'],
        ['age','risk'],
        ['risk'],
        ['risk'],
        ['age'],
        ['age']    
        ]


n = diabetic.shape[0]

population_list = [
        [True for _ in range(n)],
        [True for _ in range(n)],
        [True for _ in range(n)],
        [True for _ in range(n)],
        [True for _ in range(n)],
        [True for _ in range(n)],
        diabetic.trt == 1,
        diabetic.trt == 0,
        diabetic.trt == 1,
        diabetic.trt == 0  
        ]

sample_size_list = [
        n,
        100,
        n,
        100,
        n,
        n,
        197,
        197,
        197,
        197        
        ]


    
def get_p_values(pop, ss, vrs):
    p_lin_con = 0
    p_gau_con = 0
    p_gau_gau = 0 
    p_linfis_con = 0 
    p_cph = 0
    s_diabetic = diabetic[pop]  # select population

#    print('X', X )


    for repetition in range(num_repetitions):
        print(repetition)
        
        ss_diabetic = s_diabetic.sample(ss)
        X = np.array(ss_diabetic[vrs]).astype(np.float64)
        
        
        z = ss_diabetic.time.values
        d = ss_diabetic.status.values
        
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='lin', kernel_z='con', num_bootstrap_statistics=B)
    #    print('lin, con', p, v)
        p_lin_con +=p
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='con', num_bootstrap_statistics=B)
    #    print('gau, con', p, v)
        p_gau_con += p
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='gau', kernel_z='gau', num_bootstrap_statistics=B)
    #    print('gua, gau', p, v)
        p_gau_gau += p    
        v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='con', num_bootstrap_statistics=B)
    #    print('linfis, con', p, v)
        p_linfis_con += p
    #    v, p = wild_bootstrap_LR.wild_bootstrap_test_logrank_covariates(x=X, z=z, d=d, kernel_x='linfis', kernel_z='gau', num_bootstrap_statistics=B)
    ##    print('linfis, gau', p, v)
    #    p_linfis_gau += p
        
    #    print('CPH test', CPH_test(X, z, d))
        p_cph +=  CPH_test(X, z, d)


#    print(p_lin_con , p_gau_con , p_gau_gau , p_linfis_con,p_cph)
    
    p_value_dict = {
            'p_cph':p_cph/num_repetitions,
            'p_linfis_con':p_linfis_con/num_repetitions, 
            'p_lin_con':p_lin_con/num_repetitions , 
            'p_gau_con':p_gau_con/num_repetitions , 
            'p_gau_gau':p_gau_gau/num_repetitions , 
            }

    p_value_list = list(p_value_dict.values())
    p_value_list = [ '%.3f' % elem for elem in p_value_list ]
    p_value_list = [ss] + p_value_list
    
    print(vrs, )
    print(*p_value_list, sep = ' &  ')


get_p_values(diabetic.trt == 1, 197, ['risk'])
#name = 'pickles/diabetic_subsampling_B_' + str(B) + '_rep_' + str(rep) +'.pickle'
#with open(name, 'wb') as loc:
#    pickle.dump(p_value_dict, loc)

