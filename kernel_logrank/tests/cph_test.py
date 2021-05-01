import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def cph_test(X, z, d):
    """
    @param X: 2d numpy array
    @param z: 1d numpy array
    @param d: 1d numpy array
    @return: p-value of the cox test
    """
    # process the data back to a dataframe
    n, p = np.shape(X)
    df = pd.DataFrame({'z': z, 'd': d})
    for j in range(p):
        df['x' + str(j)] = X[:, j]

    # apply the cph test from the CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df, duration_col='z', event_col='d', show_progress=False)
    lr_test = cph.log_likelihood_ratio_test()
    p = lr_test.p_value

    return p
