import ot
import scipy as sp
import numpy as np
import pandas as pd
import dcor
from kerpy import BrownianKernel
from kernel_logrank.utils.preprocess_data import preprocess


def transformation(X, z, d, local_state):
    '''
    @param X:
    @param z:
    @param d:
    @return:
    '''
    n, dim = np.shape(X)
    # Define labels for each individual. So individual i, X[i], z[i], d[i] has a_label[i] and b_label[i]
    original_indices = np.arange(n)
    a = local_state.permutation(original_indices)
    b = local_state.permutation(original_indices)

    # Individiual a[i] is in row i, b[i] in col i
    distance_matrix = sp.spatial.distance_matrix(X, X, p=2)
    shuffled_distance_matrix = distance_matrix[np.ix_(a, b)]
    i, num_observed_events = 0, 0

    # Construct the synthetic dataset
    synthetic_X = np.zeros((n, dim))
    synthetic_Y = np.zeros(n)
    while len(a) > 1:
        # Randomly arrange both of the vectors
        length_original = len(a)
        length_synthetic = len(b)
        row = np.where(a == i)[0][0]

        if (d[i] == 1) & (length_synthetic > 1):
            p = np.ones(length_original) / length_original
            q = np.ones(length_synthetic) / length_synthetic
            G0 = ot.emd(p, q, shuffled_distance_matrix)

            # The conditional distribution is given by the index_original-th row of the coupling matrix
            conditional_prob_vector = G0[row, :] * length_original
            individual_synthetic = local_state.choice(b, p=conditional_prob_vector)
            col = np.where(b == individual_synthetic)[0][0]

            # Remove the chosen element from synthetic at risk
            synthetic_X[num_observed_events] = X[individual_synthetic]
            synthetic_Y[num_observed_events] = z[i]
            num_observed_events += 1

            b = np.delete(b, col)
            shuffled_distance_matrix = np.delete(shuffled_distance_matrix, col, axis=1)
        a = np.delete(a, row)
        shuffled_distance_matrix = np.delete(shuffled_distance_matrix, row, axis=0)
        i += 1

    # Deal with left over events
    for num, i in enumerate(b):
        synthetic_X[num_observed_events + num] = X[i]
        synthetic_Y[num_observed_events + num] = z[n - 1]

    return synthetic_X, synthetic_Y

def hsic_based_permutation_test(X, Y, num_permutations, local_state, verbose):
    '''
    @param X:
    @param Y:
    @return:
    '''
    # Define a list of statistics
    n, dim = np.shape(X)
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
    if verbose:
        print('hsic is ', hsic * 4 / n ** 2)
        print('dcor is', dcor.distance_covariance_sqr(X, Y[:, np.newaxis]))

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
    p_val = (num_permutations - k + 2) / (num_permutations + 1)

    return p_val

def opt_hsic(X, z, d, seed=1, num_permutations=1999, verbose=False, return_synthetic_data=False):
    """
    @param return_synthetic_data: Boolean if you want to access the synthetic datasets
    @param X: 2d numpy array
    @param z: 1d numpy array
    @param d: 1d numpy array
    @param seed: integer determining the seed of the permutations for reproducibility
    @param num_permutations: integer determining the number of permutations used
    @param verbose:
    @return:
    """
    # Preprocess the data, which includes sorting the data in order of increasing time
    n, dim = np.shape(X)
    local_state = np.random.RandomState(seed=seed)
    X, z, d = preprocess(X, z, d, standardize_z=True)

    # Do the transformation
    synthetic_X, synthetic_Y = transformation(X, z, d, local_state)

    p_val = hsic_based_permutation_test(synthetic_X, synthetic_Y, num_permutations, local_state, verbose)

    if return_synthetic_data:
        result = p_val, synthetic_X, synthetic_Y
    else:
        result = p_val

    return result


if __name__ == '__main__':
    # Generate some data
    from kernel_logrank.utils.generate_synthetic_data import generate_synthetic_data
    X, z, d = generate_synthetic_data(n=5)

    # Run the test
    p = opt_hsic(X, z, d, verbose=True)

