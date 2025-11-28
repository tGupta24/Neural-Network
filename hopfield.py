def hopfield_train(patterns):
    """
    patterns : list of bipolar patterns (+1/-1)
    """
    n = len(patterns[0])
    W = np.zeros((n, n))

    for p in patterns:
        p = p.reshape(n, 1)
        W += p @ p.T

    np.fill_diagonal(W, 0)  # no self-connection
    return W

def hopfield_recall(W, pattern, steps=10):
    x = pattern.copy()
    for _ in range(steps):
        for i in range(len(x)):
            net = np.dot(W[i], x)
            x[i] = 1 if net >= 0 else -1
    return x

# Example
patterns = [np.array([1, -1, 1, -1])]
W = hopfield_train(patterns)
print("Weight Matrix:\n", W)
