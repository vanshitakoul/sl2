# -*- coding: utf-8 -*-
"""perceptron.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G_lqKgWk129iuSUi9Jq8IgJnHcfAUaz1

basic working of perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Select only two classes: Setosa (0) and Versicolor (1)
X = X[Y != 2]
Y = Y[Y != 2]

# Convert class labels: 0 → -1 (Setosa), 1 → 1 (Versicolor)
Y = np.where(Y == 0, -1, 1)

# Use only two features: sepal length and sepal width
X = X[:, :2]


# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize weights and bias
W = np.zeros(X.shape[1])
b = 0
learning_rate = 0.1

# Perceptron Training
for _ in range(20):
    for i in range(len(X)):
        y_pred = np.sign(np.dot(W, X[i]) + b)
        if y_pred != Y[i]:
            W += learning_rate * Y[i] * X[i]
            b += learning_rate * Y[i]

# Split data by class for plotting
X_setosa = X[Y == -1]
X_versicolor = X[Y == 1]

# Plot data points
plt.scatter(X_setosa[:, 0], X_setosa[:, 1], color='red', label="Setosa 🌸")
plt.scatter(X_versicolor[:, 0], X_versicolor[:, 1], color='blue', label="Versicolor 🌿")

# Plot decision boundary: W1*x + W2*y + b = 0 => y = -(W1/W2)*x - (b/W2)
x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_values = -(W[0] / W[1]) * x_values - (b / W[1])
plt.plot(x_values, y_values, 'k-', label="Decision Boundary")

plt.xlabel("Sepal Length (standardized)")
plt.ylabel("Sepal Width (standardized)")
plt.legend()
plt.title("Perceptron on Iris Dataset (Setosa vs Versicolor)")
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

class ART1:
    def __init__(self, input_size, rho=0.7):
        self.input_size = input_size
        self.rho = rho
        self.weights = np.random.rand(input_size, 1)
        self.output = None

    def reset(self):

        self.weights = np.random.rand(self.input_size, 1)

    def learn(self, X, max_epochs=100):

        self.reset()
        epochs = 0

        for epoch in range(max_epochs):
            for x in X:

                x = x / np.linalg.norm(x)


                response = np.dot(self.weights.T, x)


                if response >= self.rho:

                    self.weights = self.weights + self.rho * (x - self.weights)
                else:

                    self.weights = x

            epochs += 1
            if epochs % 10 == 0:
                print(f"Epoch {epochs} completed")

        print("Training finished")

    def classify(self, X):

        results = []
        for x in X:

            x = x / np.linalg.norm(x)

            # Compute the response
            response = np.dot(self.weights.T, x)
            if response >= self.rho:
                results.append(1)
            else:
                results.append(0)
        return results


if __name__ == "__main__":

    X_train = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1]
    ])


    art1 = ART1(input_size=3, rho=0.7)


    art1.learn(X_train, max_epochs=100)


    X_test = np.array([
        [1, 0, 0],
        [0, 1, 1]
    ])


    predictions = art1.classify(X_test)
    print(f"Predictions: {predictions}")


    plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Test Data')
    plt.title('ART1 Training and Testing Data')
    plt.legend()
    plt.show()

import numpy as np

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR inputs and expected outputs
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases
np.random.seed(1)  # for reproducibility

# Input layer (2) -> Hidden layer (2)
weights_input_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(1, 2)

# Hidden layer (2) -> Output layer (1)
weights_hidden_output = np.random.rand(2, 1)
bias_output = np.random.rand(1, 1)

# Training loop
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    # --- FORWARD PASS ---
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # --- BACKPROPAGATION ---
    error = y - final_output
    output_delta = error * sigmoid_derivative(final_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # --- WEIGHTS & BIASES UPDATE ---
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# --- FINAL OUTPUT ---
print("Final Output after training:\n", final_output.round())