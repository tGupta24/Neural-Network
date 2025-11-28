import numpy as np

def adaline_train(X, T, w, b, lr=0.1, epochs=20):
    """
    ADALINE training using Delta Rule
    X : input array (n_samples × n_features)
    T : target array
    w : weights
    b : bias
    lr : learning rate
    """
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        for i in range(len(X)):
            y_in = np.dot(w, X[i]) + b  # activation = identity
            error = T[i] - y_in

            # update rule
            w = w + lr * error * X[i]
            b = b + lr * error

            print(f"Input:{X[i]} Target:{T[i]} y_in:{y_in:.3f} Error:{error:.3f} Updated w:{w}, b:{b}")

    return w, b

# Example: AND Gate (Polar form)
X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
T = np.array([1, -1, -1, -1])
w = np.array([0.2, -0.1])
b = 0

w, b = adaline_train(X, T, w, b)
print("\nFinal Weights:", w, "Bias:", b)
