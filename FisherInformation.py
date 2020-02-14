import numpy as np
import pandas as pd
from multiprocessing import Pool
import pickle


def inv_inf_matrix(sorted_X,sorted_z,sorted_d,print_score=False):
    n=np.shape(sorted_X)[0]
    p=np.shape(sorted_X)[1]
    
    #Define AR_matrix[i,:] to be the vector of indicators who are at risk at the i-th event time.
    AR_matrix=np.triu(np.ones(n))
    
    #Define S0[i] to be the number of individuals at risk at the i-th event.
    S0=n-np.arange(n)
#    print('S0',S0)
    
    #Define S1[i,:] to be the sum of the covariates at risk at time i  
    S1=np.matmul(AR_matrix,sorted_X)
#    print('S1',S1)
    
    
    #Define S2[i,:,:] to be the sum of outer products of the covariates that at risk at time i.
    sorted_X_expanded=np.zeros((n,p,p))
    for i in range(n):
        sorted_X_expanded[i,:,:]=np.outer(sorted_X[i,:],sorted_X[i,:])

    S2=np.zeros((n,p,p))
    S2[n-1,:,:]=sorted_X_expanded[n-1,:,:]
    for i in range(n-1):
        S2[n-2-i,:,:]=S2[n-1-i,:,:]+sorted_X_expanded[n-2-i]
#    print('s2',S2)
    
    
    #Define the score vector
    U=np.zeros(p)
    for i in range(n):
        if sorted_d[i]==1:
            U+=sorted_X[i,:]-S1[i,:]/S0[i]
    
    # Define V as in ....
    V=np.zeros((n,p,p))
    for i in range(n):
        if sorted_d[i]==1:
            V[i,:,:]=S2[i,:,:]/S0[i]-np.outer(S1[i,:]/S0[i],S1[i,:]/S0[i])

  
    information_matrix=np.zeros((p,p))
    for i in range(n):
        information_matrix+=V[i,:,:]
    try:
        inverse_information_matrix=np.linalg.inv(information_matrix)
#        print('yes')
    except:
        print('the pseudoinverse was used')
        inverse_information_matrix=np.linalg.pinv(information_matrix)

    #The cox score is computed and can be checked to equal LIN_FIS_LR
    if print_score:
        print('score test',np.inner(U,np.matmul(inverse_information_matrix,U)))

    return(inverse_information_matrix)

