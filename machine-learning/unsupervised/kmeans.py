"""
Topic: K-Means Clustering
Description: Unsupervised learning to group unlabeled data into 'k' clusters.
Beginner Explanation: Grouping customers based on similarity without knowing beforehand what the groups are. 'K' is the number of groups you want to find.
"""
from sklearn.cluster import KMeans
import numpy as np

# Dummy Data: [Annual Income (k$), Spending Score (1-100)]
X = np.array([[15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [18, 6]])

model = KMeans(n_clusters=2, random_state=42)
clusters = model.fit_predict(X)

print("--- Expected Outputs ---")
print("Cluster Centers:", model.cluster_centers_)
print("Assigned Clusters for each point:", clusters)

# Interview Question:
# Q: How do you choose the right value for 'K' in K-Means?
# A: Using the "Elbow Method", where you plot the Within-Cluster Sum of Squares (WCSS) against different K values and look for the 'elbow' point.
