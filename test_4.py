# Day 4: Exploring Linear Regression

import numpy as np
import matplotlib.pyplot as plt

# Simulated data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3.7, 6.1, 7.8, 10.5])

# Visualize the data
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of Simulated Data')
plt.show()

# Linear regression
def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

# Initial parameters
theta_0 = 1
theta_1 = 1

# Prediction before training
y_pred = predict(X, theta_0, theta_1)
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Initial Linear Regression')
plt.show()

# Cost function
def compute_cost(y_pred, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum((y_pred - y)**2)

initial_cost = compute_cost(y_pred, y)
print(f"Initial Cost: {initial_cost}")

# Gradient descent
def gradient_descent(X, y, theta_0, theta_1, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        y_pred = predict(X, theta_0, theta_1)
        theta_0 -= (learning_rate / m) * np.sum(y_pred - y)
        theta_1 -= (learning_rate / m) * np.sum((y_pred - y) * X)
    return theta_0, theta_1

# Training the model
learning_rate = 0.01
iterations = 1000
theta_0, theta_1 = gradient_descent(X, y, theta_0, theta_1, learning_rate, iterations)

# Prediction after training
y_pred = predict(X, theta_0, theta_1)
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression After Training')
plt.show()

final_cost = compute_cost(y_pred, y)
print(f"Final Cost: {final_cost}")
print(f"Optimal Parameters: theta_0 = {theta_0}, theta_1 = {theta_1}")
