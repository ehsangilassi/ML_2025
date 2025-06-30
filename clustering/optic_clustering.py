import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.datasets import make_blobs

# Create Artifitial data
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)


plt.scatter(X[:, 0], X[:, 1])
plt.show()

# DBSCAN
dbscan_cluster = DBSCAN(eps=1.5, min_samples=5)
# Increase eps to 1.5
dbscan_cluster.fit(X)
labels_dbscan = dbscan_cluster.labels_
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='rainbow', s=20)
plt.title('OPTICS Clustering Plot')
plt.xlabel('feat1')
plt.ylabel('feat2')
plt.grid(True)
plt.tight_layout()
plt.show()


""" OPTICS """
# OPTICS
optics_model = OPTICS(min_samples=10, xi=0.25, min_cluster_size=0.05)
optics_model.fit(X)

# labels
labels = optics_model.labels_
reachability = optics_model.reachability_[optics_model.ordering_]
space = np.arange(len(X))
ordering = optics_model.ordering_

#  Reachability Plot
plt.figure(figsize=(10, 5))
plt.bar(space, reachability, color='k')
plt.xlabel('data points ordered')
plt.ylabel('Reachability Distance')
plt.title('OPTICS Reachability Plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=20)
plt.title('OPTICS Clustering Plot')
plt.xlabel('feat1')
plt.ylabel('feat2')
plt.grid(True)
plt.tight_layout()
plt.show()
