"""
Topic: Model Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)
Beginner Explanation: 
- Accuracy: Overall correctness.
- Precision: Out of all positive PREDICTIONS, how many were actually positive? (Focus on false positives)
- Recall: Out of all ACTUAL positives, how many did we predict correctly? (Focus on false negatives)
- F1 Score: The balance between Precision and Recall.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# True labels vs Predicted labels
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

print("--- Expected Outputs ---")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall: {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")

# Interview Question:
# Q: When is Accuracy a bad metric?
# A: When dealing with Highly Imbalanced Data (e.g., 99% healthy, 1% sick). The model could just predict 'healthy' every time and get 99% accuracy but fail to detect any sick patients. Use F1-Score instead.
