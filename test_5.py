# Day 5: Embracing the Power of Neural Networks

import numpy as np
import matplotlib.pyplot as plt

# Simulated data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 5 * X + np.random.randn(100, 1)

# Visualize the data
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of Simulated Data')
plt.show()

# Neural network architecture
input_size = 1
hidden_size = 10
output_size = 1

# Initializing weights and biases
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward propagation
def forward_propagation(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    return output

# Mean Squared Error loss function
def compute_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Training the neural network
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    y_pred = forward_propagation(X)
    
    # Backpropagation
    loss = compute_loss(y_pred, y)
    output_error = y_pred - y
    delta_output = output_error
    hidden_error = delta_output.dot(weights_hidden_output.T)
    delta_hidden = hidden_error * (sigmoid(y_pred) * (1 - sigmoid(y_pred)))
    
    # Update weights and biases
    weights_hidden_output -= learning_rate * (sigmoid(X).T.dot(delta_output))
    bias_output -= learning_rate * np.sum(delta_output, axis=0)
    weights_input_hidden -= learning_rate * (X.T.dot(delta_hidden))
    bias_hidden -= learning_rate * np.sum(delta_hidden, axis=0)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Plotting the learned curve
y_learned = forward_propagation(X)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_learned, color='red', label='Learned Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Neural Network Regression')
plt.legend()
plt.show()
