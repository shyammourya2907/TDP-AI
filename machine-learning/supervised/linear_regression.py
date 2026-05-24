"""
Topic: Linear Regression
Description: A simple supervised learning algorithm used to predict a continuous numerical value.
Beginner Explanation: Imagine drawing a straight line through a scatterplot of points that best represents the trend. That's Linear Regression!
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Dummy Data: Years of Experience vs Salary
X = np.array([[1], [2], [3], [4], [5]])  # Years of Experience
y = np.array([45000, 50000, 60000, 65000, 70000])  # Salary in $

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 3. Make Predictions
predictions = model.predict(X)

print("--- Expected Outputs ---")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", predictions)

# Interview Question:
# Q: What are the assumptions of Linear Regression?
# A: Linearity, independence, homoscedasticity (constant variance of errors), and normality of error distribution.
