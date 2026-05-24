"""
Topic: Principal Component Analysis (PCA)
Description: A dimensionality reduction technique.
Beginner Explanation: Shrinking data from 3D to 2D (or keeping the most important features) while losing as little information as possible.
"""
from sklearn.decomposition import PCA
import numpy as np

# 3D Data
X = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]])

# Reduce to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("--- Expected Outputs ---")
print("Original Shape:", X.shape)
print("Reduced Shape:", X_reduced.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Interview Question:
# Q: When would you use PCA?
# A: To reduce the number of features to speed up training, reduce noise, and visualize high-dimensional data.
