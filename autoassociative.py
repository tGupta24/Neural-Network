def auto_associative_hebb(X):
    n = len(X[0])
    W = np.zeros((n,n))

    for p in X:
        p = p.reshape(n,1)
        W += p @ p.T

    np.fill_diagonal(W,0)
    return W
