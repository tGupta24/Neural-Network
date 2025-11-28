def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    return x*(1-x)

def backprop_train(X, T, hidden=2, lr=0.1, epochs=5000):
    n_samples, n_features = X.shape
    n_output = T.shape[1]

    W1 = np.random.randn(n_features, hidden)
    B1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, n_output)
    B2 = np.zeros(n_output)

    for ep in range(epochs):
        # Forward pass
        Z1 = sigmoid(X @ W1 + B1)
        Z2 = sigmoid(Z1 @ W2 + B2)

        # Backprop
        error = T - Z2
        dZ2 = error * dsigmoid(Z2)

        dW2 = Z1.T @ dZ2
        dB2 = np.sum(dZ2, axis=0)

        dZ1 = dZ2 @ W2.T * dsigmoid(Z1)
        dW1 = X.T @ dZ1
        dB1 = np.sum(dZ1, axis=0)

        # Update
        W1 += lr * dW1
        B1 += lr * dB1
        W2 += lr * dW2
        B2 += lr * dB2

    return W1, B1, W2, B2
