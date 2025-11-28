def hetero_associative_hebb(X, Y):
    """
    X = input patterns (n_samples × n)
    Y = target patterns (n_samples × m)
    """
    X = np.array(X)
    Y = np.array(Y)

    W = Y.T @ X  # m × n
    return W

def recall(W, x):
    y = np.dot(W, x)
    return np.where(y >= 0, 1, -1)
