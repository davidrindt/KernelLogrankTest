#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:42:59 2019

@author: david.rindt
"""

"""
Created on Mon Dec 24 16:13:32 2018
@author: David
"""


import ot
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dcor
from multiprocessing import Pool
import pickle
from kerpy import BrownianKernel
from scipy import stats
import sys
import warnings

def opt_hsic(x,z,d,seed,alpha=0.05,num_permutations=1999,verbose=False):
    n=np.shape(x)[0]
    local_state=np.random.RandomState(seed=seed)
    #Sort the data in order of increasing time.
    indices=np.arange(n)
    z, list2= (np.array((t)) for t in zip(*sorted(zip(z,indices))))
    d=d[list2]
    x=x[list2]
    # INITIALIZE AT RISK SETS
    original_atrisk=x
    synthetic_atrisk=x
    X=np.zeros(np.shape(x))
    Y=np.zeros(n)
    num_observed_events=0
    #CONSTRUCT THE TRANSFORMED DATASET
    for i in range(n):
        #randomly arange both of the vectors
        original_atrisk=np.random.permutation(original_atrisk)
        synthetic_atrisk=np.random.permutation(synthetic_atrisk)
        lenor=len(original_atrisk)
        lensyn=len(synthetic_atrisk)
        #find the index of the event covariate
        index_original=np.where(original_atrisk==x[i])[0]
        if d[i]==1:
            #if it's observed, compute the coupling
            p=np.ones(lenor)/lenor
            q=np.ones(lensyn)/lensyn
            G0 = ot.emd(p, q, sp.spatial.distance_matrix(original_atrisk,synthetic_atrisk,p=2))
            #the conditional distribution is given by the index_original-th row of the coupling matrix
            conditional_prob_vector=G0[index_original,:]*lenor
            
            try:
                index_tilde_x=local_state.choice(np.arange(len(synthetic_atrisk)),p=conditional_prob_vector[0])
#                print('succes')
            except:
                print('fail')
                if True:
                    index_tilde_x=local_state.choice(np.arange(len(synthetic_atrisk)))
                else:
                    sys.exit('something went wrong')
                
            #remove the chosen element from synthetic at risk
            tilde_x=synthetic_atrisk[index_tilde_x]
            synthetic_atrisk=np.delete(synthetic_atrisk,index_tilde_x,axis=0)
            X[num_observed_events]=tilde_x
            Y[num_observed_events]=z[i]
            num_observed_events+=1
        #remove the orignal element from the at original at risk set
        original_atrisk=np.delete(original_atrisk,index_original,axis=0)
    
    #deal with left over events   
    for i in range(len(synthetic_atrisk)):
        X[num_observed_events+i]=synthetic_atrisk[i]
        Y[num_observed_events+i]=z[n-1]

    #define a list of statistics
    statistic_list=np.zeros(num_permutations+1)
    
    #Define the kernels, kernel matriecs
    k=BrownianKernel.BrownianKernel()
    k.alpha=1
    l=BrownianKernel.BrownianKernel()
    l.alpha=1
    Kx=k.kernel(X)
    Ky=l.kernel(Y[:,np.newaxis])
    prod_Kx_H=Kx-np.outer( Kx @ np.ones(n) , np.ones(n)  )/n
    HKxH=prod_Kx_H-np.outer(np.ones(n), np.transpose(np.ones(n)) @ prod_Kx_H)/n
    hsic=np.sum(np.multiply(HKxH,Ky))
    #Check if the HSIC computation is alright.
    if verbose==True:
        print('hsic is ', hsic*4/n**2)
        print('dcor is', dcor.distance_covariance_sqr(x,z))
    statistic_list[0]=hsic
    
    counting=np.arange(n)
    #do a permutation test with num_permutations permutations
    
    for permutation in range(num_permutations):
        a=local_state.permutation(counting)
        permuted_Ky=Ky[np.ix_(a,a)]
        statistic_list[permutation+1]=np.sum(np.multiply(HKxH,permuted_Ky))

    vec=pd.Series(statistic_list)
    vec=vec.sample(frac=1).rank(method='first')
    k=vec[0]
    
    return((num_permutations-k+2)/(num_permutations+1))
