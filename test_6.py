# Day 6: Exploring Deep Learning with Keras

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

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

# Creating a deep learning model
model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compiling the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Plotting the training loss
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Training Loss Over Time')
plt.show()

# Predictions with the trained model
y_pred = model.predict(X)

# Plotting the learned curve
plt.scatter(X, y, color='blue', label='Actual Data')
plt.scatter(X, y_pred, color='red', label='Predicted Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Deep Learning Regression')
plt.legend()
plt.show()
