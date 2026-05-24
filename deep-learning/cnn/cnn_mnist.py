"""
Topic: Convolutional Neural Network (CNN) - Image Classification
Beginner Explanation: CNNs look at images through a sliding window (filter) to detect patterns like edges, shapes, and eventually faces or objects.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)), # Filters
    MaxPooling2D((2,2)), # Pooling (downsampling)
    Flatten(), # Flatten 2D to 1D
    Dense(10, activation='softmax') # Output Layer for 10 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--- Expected Outputs ---")
model.summary()

# Interview Question:
# Q: What does MaxPooling do?
# A: It reduces the spatial dimensions (width and height) of the image representation, keeping the most important features and reducing computation.
