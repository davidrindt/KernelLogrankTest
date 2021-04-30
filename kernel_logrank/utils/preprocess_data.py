import numpy as np

def preprocess(X, z, d, standardize_z=True):
    """
    @param X:
    @param z:
    @param d:
    @param standardize_z: Boolean indicating weather or not to standardize z
    @return:
    """

    # Sort the data in order of increasing time
    n, p = np.shape(X)
    indices = np.arange(n)
    z, sorted_indices = (np.array(t) for t in zip(*sorted(zip(z, indices))))
    d = d[sorted_indices]
    X = X[sorted_indices]

    # Standardize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    if standardize_z:
        z = (z - z.mean()) / z.std()

    return X, z, d