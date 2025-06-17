import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

kmedoids = KMedoids(n_clusters=4, random_state=42, metric='euclidean')
kmedoids.fit(X)

labels = kmedoids.labels_
medoids = kmedoids.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids')
plt.title("K-Medoids Clustering")
plt.legend()
plt.grid(True)
plt.show()
