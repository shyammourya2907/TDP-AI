"""
Topic: Dropout (Regularization)
Beginner Explanation: During training, Dropout randomly turns off some neurons. This forces the remaining neurons to learn robust features and prevents the network from memorizing the data (Overfitting).
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_dim=10),
    Dropout(0.5), # 50% chance a neuron gets turned off
    Dense(1, activation='sigmoid')
])

print("Dropout Layer added to prevent overfitting!")
