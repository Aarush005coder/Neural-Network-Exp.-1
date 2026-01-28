import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=50,
    hypercube=False,
    class_sep=10
)

y = np.where(y == 0, -1, 1)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=70)
plt.title("Training Data")
plt.show()

def step(z):
    return 1 if z >= 0 else -1

def perceptron(X, y, lr=1.0, epochs=100):
    X = np.insert(X, 0, 1, axis=1)
    weights = np.zeros(X.shape[1])

    slopes = []
    intercepts = []

    print("PERCEPTRON TRAINING")
    for epoch in range(epochs):
        errors = 0
        print(f"\nEpoch {epoch+1}")

        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i]

            activation = np.dot(weights, x_i)
            y_hat = step(activation)

            print("x :", x_i)
            print("w :", weights)
            print("w.T * x :", activation)

            if y_i != y_hat:
                weights = weights + lr * y_i * x_i
                errors += 1

            slopes.append(-weights[1] / weights[2])
            intercepts.append(-weights[0] / weights[2])

        if errors == 0:
            print("\nConverged early ✔")
            break
          
    print("\nFinal weights:", weights)
    return weights, slopes, intercepts

weights, m, b = perceptron(X, y)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X[:,0], X[:,1], c=y, cmap='winter', s=70)

x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
line, = ax.plot(x_vals, m[0]*x_vals + b[0], 'r-', linewidth=2)

def update(i):
    line.set_ydata(m[i]*x_vals + b[i])
    ax.set_title(f"Perceptron Learning | Step {i+1}")
    return line,

anim = FuncAnimation(
    fig,
    update,
    frames=len(m),
    interval=120,
    repeat=False
)

plt.tight_layout()
HTML(anim.to_jshtml())
