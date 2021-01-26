import numpy as np
import pandas as pd
from multiprocessing import Pool
import pickle
#import scipy.spatial
from lifelines import CoxPHFitter
cph = CoxPHFitter()



def CPH_test(x,z,d,alpha=0.05,print_score=False):
    n=len(z)
    p=np.shape(x)[1]
    df=pd.DataFrame({'z':z,'d':d})
    for j in range(p):
        df['x'+str(j)]=x[:,j]
    cph.fit(df, duration_col='z', event_col='d', show_progress=False)
    try:
        a=cph.log_likelihood_ratio_test()
        pvalue=a.p_value
    except:
        print('the likelihood ratio test failed')
        pvalue=1
    if print_score:
        print('n is',n,'cph',a.test_statistic)
    return(pvalue)

