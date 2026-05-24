"""
Topic: Cross-Validation (k-Fold)
Beginner Explanation: Instead of splitting data once (train/test), we split it 'k' times and train 'k' models to ensure our model performs well on all parts of the data.
"""
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier()

# 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("--- Expected Outputs ---")
print("Scores for each fold:", scores)
print("Average CV Accuracy:", scores.mean())
