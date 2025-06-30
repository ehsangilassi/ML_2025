from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from numpy import random
from sklearn.datasets._samples_generator import make_blobs
from pandas import read_csv
from utility import DATA_PATH
""" Test Data """
random.seed(1)
x, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.5)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

sc = SpectralClustering(n_clusters=4).fit(x)
print(sc)

labels = sc.labels_
plt.scatter(x[:, 0], x[:, 1], c=labels)
plt.show()

f = plt.figure()
f.add_subplot(2, 2, 1)
for i in range(2, 6):
    sc = SpectralClustering(n_clusters=i).fit(x)
    f.add_subplot(2, 2, i - 1)
    plt.scatter(x[:, 0], x[:, 1], s=5, c=sc.labels_, label="n_cluster-" + str(i))
    plt.legend()

plt.show()


""" Spiral Data """

file_name = 'spiral.csv'
df = read_csv(DATA_PATH + file_name)

df.plot.scatter(x="X", y= "Y")
plt.show()

sc = SpectralClustering(n_clusters=3).fit(df)
print(sc)

labels = sc.labels_
plt.scatter(df["X"], df["Y"], c=labels)
plt.show()
