import numpy as np
import pandas as pd
from multiprocessing import Pool
import pickle
#import scipy.spatial
from lifelines import CoxPHFitter
cph = CoxPHFitter()
from lifelines.statistics import chisq_test


def CPH_test(x,z,d,alpha=0.05,print_score=False):
    n, p = np.shape(x)
    df=pd.DataFrame({'z':z,'d':d})
    for j in range(p):
        df['x'+str(j)]=x[:,j]
    cph = CoxPHFitter()
    cph.fit(df, duration_col='z', event_col='d', show_progress=False)
    test_stat, degrees_freedom, minus_log2_p = cph._compute_likelihood_ratio_test()
    # print('first try', chisq_test(test_stat, degrees_freedom))
    p = cph.summary['p'].to_numpy()[0]
    return p

if __name__ == '__main__':

    # Generate some data
    local_state = np.random.RandomState(1)
    n = 200
    dim = 10
    x = local_state.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=n)
    c = np.zeros(n)
    rowsum = np.sum(x, axis=1)
    for v in range(n):
        c[v] = local_state.exponential(np.exp((rowsum[v] / 8)))
    t = local_state.exponential(.6, size=n)
    d = np.int64(c > t)
    z = np.minimum(t, c)

    # Run the test
    print(CPH_test(x, z, d))
