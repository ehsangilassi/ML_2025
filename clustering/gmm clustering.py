from pandas import read_csv, DataFrame
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from numpy import unique

from utility import DATA_PATH
file_name = 'unbalanced.txt'
data = read_csv(DATA_PATH + file_name, sep=' ', names=["spending", "income"])
print(data)

data.plot.scatter(x="spending", y="income", s=20)
plt.title(" Data Distribution")
plt.xlabel("Spending")
plt.ylabel("Income")
plt.show()

gmm = GaussianMixture(n_components=8)
gmm.fit(data)

# predictions from gmm
labels = gmm.predict(data)

data['cluster'] = labels
number_of_cluster = len(data['cluster'].unique())
color = ['blue', 'green', 'cyan', 'black', 'red', 'brown', 'purple', 'pink']
for k in range(0, number_of_cluster):
    cluster = data[data["cluster"] == k]
    plt.scatter(cluster["spending"], cluster["income"], c=color[k])
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Spending")
plt.ylabel("Income")
plt.show()



# How to find best n_component:
# AIC, BIC
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

n_components = range(1, 8)
aic = []
bic = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=0)
    gmm.fit(X)
    aic.append(gmm.aic(X))
    bic.append(gmm.bic(X))

plt.plot(n_components, aic, label='AIC', marker='o')
plt.plot(n_components, bic, label='BIC', marker='s')
plt.xlabel('Clusters')
plt.ylabel('value')
plt.title('Find best n_component in GMM ')
plt.legend()
plt.grid(True)
plt.show()



# TEST on data
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# predictions from gmm
labels = gmm.predict(X)
X = DataFrame(X, columns=['feat1', 'feat2'])
X['cluster'] = labels
number_of_cluster = len(X['cluster'].unique())
color = ['blue', 'green', 'red']
for k in range(0, number_of_cluster):
    cluster = X[X["cluster"] == k]
    plt.scatter(cluster["feat1"], cluster["feat2"], c=color[k])
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("feat1")
plt.ylabel("feat2")
plt.show()
