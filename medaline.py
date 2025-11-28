import numpy as np

def madaline_train(X, T, W, B, lr=0.1, epochs=10):
    """
    W: weight matrix (hidden_units × features)
    B: bias array for hidden units
    """
    H = len(W)  # hidden units

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        for i in range(len(X)):

            # Hidden layer outputs (ADALINE units)
            y_hidden = np.dot(W, X[i]) + B

            # Activation = sign
            y_hidden_act = np.where(y_hidden >= 0, 1, -1)

            # Final output
            y_out = 1 if np.sum(y_hidden_act) >= 0 else -1

            error = T[i] - y_out

            # Update rule: update only the unit causing error
            if error != 0:
                for h in range(H):
                    W[h] += lr * (T[i] - y_hidden_act[h]) * X[i]
                    B[h] += lr * (T[i] - y_hidden_act[h])

            print(f"Input:{X[i]} Hidden:{y_hidden_act} Output:{y_out} Error:{error}")
            print("Updated W:", W, "Updated B:", B)

    return W, B
