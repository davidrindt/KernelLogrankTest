import ot
import scipy as sp
import numpy as np
import pandas as pd
import dcor
from kerpy import BrownianKernel
from kernel_logrank.utils.preprocess_data import preprocess


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
    # Preprocess the data, which includes sorting the data in order of increasing time
    n, _ = np.shape(X)
    local_state = np.random.RandomState(seed=seed)
    X, z, d = preprocess(X, z, d, standardize_z=True)

    # Define labels for each individual. So individual i, X[i], z[i], d[i] has a_label[i] and b_label[i]
    original_indices = np.arange(n)
    a_labels = local_state.permutation(original_indices)
    new_row_from_old_row = {a_labels[i]: i for i in range(n)}
    b_labels = local_state.permutation(original_indices)
    new_col_from_old_col = {b_labels[i]: i for i in range(n)}
    distance_matrix = sp.spatial.distance_matrix(X, X, p=2)

    # Let the distance_matrix[i, j] = dist(individual label i from original, individual label j from synthetic)
    new_distance_matrix = distance_matrix[np.ix_(a_labels, b_labels)]
    print(new_distance_matrix[1, 3])
    print(distance_matrix[a_labels[1], b_labels[3]])
    print(distance_matrix[10,11])
    print(new_distance_matrix[new_row_from_old_row[10], new_col_from_old_col[11]])
    # Construct the synthetic dataset
    while len(a_labels) > 1:
        # Randomly arrange both of the vectors
        length_original = len(a_labels)
        length_synthetic = len(b_labels)
        index_original = new_row_from_old_row[i]
        print('idx orig', index_original)
        if (d[i] == 1) & (length_synthetic > 1):

            p = np.ones(length_original) / length_original
            q = np.ones(length_synthetic) / length_synthetic
            print(f'p {p}, q, {q}')
            G0 = ot.emd(p, q, sp.spatial.distance_matrix(original_at_risk, synthetic_at_risk, p=2))
            # print(p, q, G0)
            # print(np.sum(q))
            # print('sum g0', np.sum(G0))

            # the conditional distribution is given by the index_original-th row of the coupling matrix
            conditional_prob_vector = G0[index_original, :] * length_original
            # print('cond prob vec', conditional_prob_vector)
            index_tilde_x = local_state.choice(np.arange(len(synthetic_at_risk)), p=conditional_prob_vector)

            # except:
            #     print('failed to do the optimal transport step')
            #     if True:
            #         index_tilde_x = local_state.choice(np.arange(len(synthetic_at_risk)))
            #     else:
            #         sys.exit('something went wrong')

            # Remove the chosen element from synthetic at risk
            tilde_x = synthetic_at_risk[index_tilde_x]
            synthetic_at_risk = np.delete(synthetic_at_risk, index_tilde_x, axis=0)
            synthetic_X[num_observed_events] = tilde_x
            synthetic_Y[num_observed_events] = z[i]
            num_observed_events += 1

        # Remove the original element from the at original at risk set
        original_at_risk = np.delete(original_at_risk, index_original, axis=0)

    # deal with left over events
    for i in range(len(synthetic_at_risk)):
        synthetic_X[num_observed_events + i] = synthetic_at_risk[i]
        synthetic_Y[num_observed_events + i] = z[n - 1]

    # define a list of statistics
    statistic_list = np.zeros(num_permutations + 1)

    # Define the kernels, kernel matrices
    k = BrownianKernel.BrownianKernel()
    k.alpha = 1
    l = BrownianKernel.BrownianKernel()
    l.alpha = 1
    Kx = k.kernel(X)
    Ky = l.kernel(synthetic_Y[:, np.newaxis])

    # Compute HSIC
    prod_Kx_H = Kx - np.outer(Kx @ np.ones(n), np.ones(n)) / n
    HKxH = prod_Kx_H - np.outer(np.ones(n), np.transpose(np.ones(n)) @ prod_Kx_H) / n
    hsic = np.sum(np.multiply(HKxH, Ky))

    # Check if the HSIC computation is correct
    if verbose == True:
        print('hsic is ', hsic * 4 / n ** 2)
        print('dcor is', dcor.distance_covariance_sqr(synthetic_X, synthetic_Y[:, np.newaxis]))

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


if __name__ == '__main__':
    # Generate some data
    from kernel_logrank.utils.generate_synthetic_data import generate_synthetic_data

    X, z, d = generate_synthetic_data()
    print('X', X)
    # Run the test
    print(opt_hsic(X, z, d, verbose=True))
    print(opt_hsic(X, z, d, verbose=True))
