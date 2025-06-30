import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

""" Plot Data """
sns.set_style('dark')
X1 = np.array([[1, 1], [3, 2], [9, 1], [3, 7], [7, 2], [9, 7], [4, 8], [8, 3], [1, 4]])
plt.figure(figsize=(6, 6))
plt.scatter(X1[:, 0], X1[:, 1], c='r')
plt.show()
""" Create numbered labels for each point """
for i in range(X1.shape[0]):
    plt.annotate(str(i), xy=(X1[i, 0], X1[i, 1]), xytext=(3, 3), textcoords='offset points')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.title('Scatter Plot of the data')
plt.xlim([0, 10]), plt.ylim([0, 10])
plt.xticks(range(10)), plt.yticks(range(10))
plt.grid()
plt.show()

""" Calculate Dendrogram"""
Z1 = linkage(X1, method='single', metric='euclidean')
Z2 = linkage(X1, method='complete', metric='euclidean')
Z3 = linkage(X1, method='average', metric='euclidean')
Z4 = linkage(X1, method='ward', metric='euclidean')

""" Plot Dendrogram """
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1), dendrogram(Z1), plt.title('Single')
plt.subplot(2, 2, 2), dendrogram(Z2), plt.title('Complete')
plt.subplot(2, 2, 3), dendrogram(Z3), plt.title('Average')
plt.subplot(2, 2, 4), dendrogram(Z4), plt.title('Ward')
plt.show()



""" Scikit-learn """
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# create data
X, y = make_blobs(n_samples=10, centers=3, random_state=42)

model = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
model.fit(X)

labels = model.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Hierarchical Clustering (Scikit-learn)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()



#  linkage matrix
linked = linkage(X, method='ward')

# Dendrogram plot
plt.figure(figsize=(8, 4))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()


""" How to find best number of clusters? """
# We can Use silhouette_score:
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np

range_n_clusters = range(2, 10)
scores = []

for n_clusters in range_n_clusters:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

# plot
plt.plot(range_n_clusters, scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.grid(True)
plt.show()

optimal_n = range_n_clusters[np.argmax(scores)]
print(f"best number of clusters with silhouette score: {optimal_n}")

