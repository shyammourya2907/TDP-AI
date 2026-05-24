"""
Topic: Recurrent Neural Networks (RNN/LSTM)
Description: Used for sequential data like Time Series or Text.
Beginner Explanation: Standard Neural Networks have no memory of the past. RNNs have a loop, allowing information to persist. LSTMs are special RNNs that can remember long-term dependencies.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build LSTM Model for Sequence Prediction
model = Sequential([
    LSTM(50, input_shape=(10, 1)), # 10 timesteps, 1 feature
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("--- Expected Outputs ---")
model.summary()

# Interview Question:
# Q: Why use LSTM instead of a basic RNN?
# A: Basic RNNs suffer from the "Vanishing Gradient Problem", making it hard to learn long-term dependencies. LSTMs have "gates" that control information flow, solving this issue.
