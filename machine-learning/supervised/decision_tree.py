"""
Topic: Decision Trees
Description: A tree-like model used for both classification and regression.
Beginner Explanation: Think of it like playing "20 Questions". The model asks a series of True/False questions to narrow down the answer.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris Dataset
data = load_iris()
X = data.data
y = data.target

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("--- Expected Outputs ---")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Interview Question:
# Q: What is a major disadvantage of Decision Trees?
# A: They are highly prone to Overfitting (memorizing the training data). Random Forests help solve this.
