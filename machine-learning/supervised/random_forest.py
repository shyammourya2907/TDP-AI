"""
Topic: Random Forest
Description: An ensemble learning method using multiple decision trees.
Beginner Explanation: Instead of asking one person (Decision Tree) for an answer, you ask a crowd of 100 people and take the majority vote. This is "Ensemble Learning".
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100) # 100 trees
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("--- Expected Outputs ---")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Interview Question:
# Q: How does a Random Forest prevent overfitting?
# A: By building multiple trees on random subsets of data and features, and averaging their predictions, it reduces variance and prevents overfitting.
