import numpy as np


def inv_inf_matrix(X, d, print_score=False):
    """
    @param X: 2d numpy array. Must be sorted in increasing time
    @param d: 1d numpy array. Must be sorted in increasing time
    @param print_score: Boolean. Indicates if something should be printed.
    @return:
    """
    n, p = np.shape(X)

    # Define AR_matrix[i,:] to be the vector of indicators who are at risk at the i-th event time.
    AR_matrix = np.triu(np.ones(n))

    # Define S0[i] to be the number of individuals at risk at the i-th event.
    S0 = n - np.arange(n)

    # Define S1[i,:] to be the sum of the covariates at risk at time i
    S1 = np.matmul(AR_matrix, X)

    # Define S2[i,:,:] to be the sum of outer products of the covariates that at risk at time i.
    x_expanded = np.zeros((n, p, p))
    for i in range(n):
        x_expanded[i, :, :] = np.outer(X[i, :], X[i, :])

    S2 = np.zeros((n, p, p))
    S2[n - 1, :, :] = x_expanded[n - 1, :, :]
    for i in range(n - 1):
        S2[n - 2 - i, :, :] = S2[n - 1 - i, :, :] + x_expanded[n - 2 - i]

    # Define the score vector
    U = np.zeros(p)
    for i in range(n):
        if d[i] == 1:
            U += X[i, :] - S1[i, :] / S0[i]

    # Define V as in ....
    V = np.zeros((n, p, p))
    for i in range(n):
        if d[i] == 1:
            V[i, :, :] = S2[i, :, :] / S0[i] - np.outer(S1[i, :] / S0[i], S1[i, :] / S0[i])

    information_matrix = np.zeros((p, p))
    for i in range(n):
        information_matrix += V[i, :, :]
    try:
        inverse_information_matrix = np.linalg.inv(information_matrix)
    except:
        print('the pseudoinverse was used')
        inverse_information_matrix = np.linalg.pinv(information_matrix)

    # The cox score is computed and can be checked to equal LIN_FIS_LR
    if print_score:
        print('score test', np.inner(U, np.matmul(inverse_information_matrix, U)))

    return inverse_information_matrix
