import numpy as np
import matplotlib.pyplot as plt

a = np.array([0.8, 0.5, 0.2], dtype=float)
epsilon = 0.2
max_iters = 20

history = [a.copy()]

print("Iteration 0:", np.round(a, 4))

for k in range(max_iters):
    a_new = a - epsilon * (np.sum(a) - a)
    a_new[a_new < 0] = 0
    history.append(a_new.copy())
    print(f"Iteration {k+1}:", np.round(a_new, 4))
    if np.allclose(a_new, a):
        print(f"\n Converged at iteration {k+1}")
        break
    a = a_new

winner = np.argmax(a) + 1
print(f"\n🏆 Winner neuron index: {winner}")

history = np.array(history)

plt.figure(figsize=(8,5))
for i in range(history.shape[1]):
    plt.plot(history[:, i], marker='o', label=f'Neuron {i+1}')
    
plt.title("MAXNET Convergence (Winner-Take-All Competition)")
plt.xlabel("Iteration")
plt.ylabel("Activation Value")
plt.legend()
plt.grid(True)
plt.show()


output

Iteration 0: [0.8 0.5 0.2]
Iteration 1: [0.66 0.3  0.  ]
Iteration 2: [0.6   0.168 0.   ]
Iteration 3: [0.5664 0.048  0.    ]
Iteration 4: [0.5568 0.     0.    ]
Iteration 5: [0.5568 0.     0.    ]



