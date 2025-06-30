from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from numpy import sort
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utility import DATA_PATH


X, _ = make_moons(n_samples=200, noise=0.05, random_state=42, shuffle=True)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1])
plt.title('Moon-shaped Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

""" Compare between DBSCAN & K-Means """
# DBSCAN clustering
dbscan = DBSCAN(eps=0.17, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
ax1.set_title('DBSCAN Clustering')

ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
ax2.set_title('K-Means Clustering')

plt.show()

""" Find best Eps """


def plot_k_distance_graph(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    distances = sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph')
    plt.show()

# Plot k-distance graph
plot_k_distance_graph(X, k=5)


epsilon = 0.15  # Chosen based on k-distance graph
min_samples = 5  # 2 * num_features (2D data)
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



""" Cluster Check in data """
# Read Data
file_name = 'unbalanced.txt'
df = read_csv(DATA_PATH + file_name, sep=' ', names=["X", "Y"])
print(df)

""" Plot Check_in Data for Tehran """
df.plot.scatter(x='X', y='Y', s=20)
plt.title('Check-in Data in Tehran')
plt.xlabel('Longitude'), plt.ylabel('Latitude')
plt.show()

#  Normalized Data:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# DBSCAN Calculation:
clustering = DBSCAN(eps=0.3, min_samples=5)
clusters = clustering.fit_predict(X_scaled)
label_of_points = clustering.labels_
n_clusters_ = len(set(label_of_points)) - (1 if -1 in label_of_points else 0)
n_noise_ = list(label_of_points).count(-1)
print('Lable of points:')
print(label_of_points)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Plot result
plt.scatter(df["X"], df["Y"], c=clusters, cmap="plasma")
plt.show()