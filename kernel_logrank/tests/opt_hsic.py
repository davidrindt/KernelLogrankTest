import ot
import scipy as sp
import numpy as np
import pandas as pd
import dcor
from kerpy import BrownianKernel
from utils.preprocess_data import preprocess


def opt_hsic(X, z, d, seed=1, num_permutations=1999, verbose=False):
    """
    @param X: 2d numpy array
    @param z: 1d numpy array
    @param d: 1d numpy array
    @param seed: integer determining the seed of the permutations for reproducibility
    @param num_permutations: integer determining the number of permutations used
    @param verbose:
    @return:
    """
    n, p = np.shape(X)
    local_state = np.random.RandomState(seed=seed)

    # Preprocess the data, which includes sorting the data in order of increasing time
    X, z, d = preprocess(X, z, d, standardize_z=True)

    # Initialize the at risk sets
    original_at_risk = X
    synthetic_at_risk = X
    X = np.zeros(np.shape(X))
    Y = np.zeros(n)
    num_observed_events = 0

    # Construct the synthetic dataset
    for i in range(n):

        # Randomly arange both of the vectors
        original_at_risk = np.random.permutation(original_at_risk)
        synthetic_at_risk = np.random.permutation(synthetic_at_risk)
        length_original = len(original_at_risk)
        length_synthetic = len(synthetic_at_risk)

        # find the index of the event covariate
        index_original = np.where(original_at_risk == X[i])[0]
        if d[i] == 1:

            p = np.ones(length_original) / length_original
            q = np.ones(length_synthetic) / length_synthetic
            G0 = ot.emd(p, q, sp.spatial.distance_matrix(original_at_risk, synthetic_at_risk, p=2))

            # the conditional distribution is given by the index_original-th row of the coupling matrix
            conditional_prob_vector = G0[index_original, :] * length_original

            try:
                index_tilde_x = local_state.choice(np.arange(len(synthetic_at_risk)), p=conditional_prob_vector[0])

            except:
                print('fail')
                if True:
                    index_tilde_x = local_state.choice(np.arange(len(synthetic_at_risk)))
                else:
                    sys.exit('something went wrong')

            # Remove the chosen element from synthetic at risk
            tilde_x = synthetic_at_risk[index_tilde_x]
            synthetic_at_risk = np.delete(synthetic_at_risk, index_tilde_x, axis=0)
            X[num_observed_events] = tilde_x
            Y[num_observed_events] = z[i]
            num_observed_events += 1

        # Remove the original element from the at original at risk set
        original_at_risk = np.delete(original_at_risk, index_original, axis=0)

    # deal with left over events
    for i in range(len(synthetic_at_risk)):
        X[num_observed_events + i] = synthetic_at_risk[i]
        Y[num_observed_events + i] = z[n - 1]

    # define a list of statistics
    statistic_list = np.zeros(num_permutations + 1)

    # Define the kernels, kernel matrices
    k = BrownianKernel.BrownianKernel()
    k.alpha = 1
    l = BrownianKernel.BrownianKernel()
    l.alpha = 1
    Kx = k.kernel(X)
    Ky = l.kernel(Y[:, np.newaxis])

    # Compute HSIC
    prod_Kx_H = Kx - np.outer(Kx @ np.ones(n), np.ones(n)) / n
    HKxH = prod_Kx_H - np.outer(np.ones(n), np.transpose(np.ones(n)) @ prod_Kx_H) / n
    hsic = np.sum(np.multiply(HKxH, Ky))

    # Check if the HSIC computation is correct
    if verbose == True:
        print('hsic is ', hsic * 4 / n ** 2)
        print('dcor is', dcor.distance_covariance_sqr(X, z))

    # Do a permutation test with num_permutations permutations
    statistic_list[0] = hsic
    counting = np.arange(n)
    for permutation in range(num_permutations):
        a = local_state.permutation(counting)
        permuted_Ky = Ky[np.ix_(a, a)]
        statistic_list[permutation + 1] = np.sum(np.multiply(HKxH, permuted_Ky))

    # Compute the rank, breaking ties at random
    vec = pd.Series(statistic_list)
    vec = vec.sample(frac=1).rank(method='first')
    k = vec[0]

    return (num_permutations - k + 2) / (num_permutations + 1)
