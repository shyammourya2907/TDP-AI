"""
Topic: Logistic Regression
Description: A supervised learning algorithm used for binary classification.
Beginner Explanation: Despite its name, Logistic Regression is used for CLASSIFICATION (e.g., Yes/No, Spam/Not Spam), not regression. It uses a Sigmoid function to output a probability between 0 and 1.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Dummy Data: Hours Studied vs Pass(1)/Fail(0)
X = np.array([[1], [2], [3], [4], [5], [6]]) 
y = np.array([0, 0, 0, 1, 1, 1])

# 2. Train the Model
model = LogisticRegression()
model.fit(X, y)

# 3. Predict new data
new_student = np.array([[3.5]])
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("--- Expected Outputs ---")
print(f"Hours Studied: {new_student[0][0]}")
print(f"Predicted Class (1=Pass, 0=Fail): {prediction[0]}")
print(f"Probability [Fail, Pass]: {probability[0]}")

# Interview Question:
# Q: Why can't we use Linear Regression for Classification?
# A: Linear regression outputs continuous values (e.g., -5, 120), whereas classification needs discrete categories (0 or 1). Logistic regression maps outputs to probabilities using the sigmoid function.
