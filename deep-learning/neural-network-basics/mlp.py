"""
Topic: Multi-Layer Perceptron (Neural Network Basics)
Description: A simple Feedforward Neural Network using Keras.
Beginner Explanation: Think of it as a complex function that learns weights to map inputs to outputs. It has input layers, hidden layers, and output layers.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Dummy Data: XOR problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Build Model
model = Sequential([
    Dense(4, input_dim=2, activation='relu'), # Hidden Layer
    Dense(1, activation='sigmoid')            # Output Layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
print("Training Model...")
model.fit(X, y, epochs=100, verbose=0)

# Predict
predictions = model.predict(X)

print("--- Expected Outputs ---")
print(predictions.round())

# Interview Question:
# Q: What is an Activation Function?
# A: It introduces non-linearity into the network, allowing it to learn complex patterns (e.g., ReLU, Sigmoid).
