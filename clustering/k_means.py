#  Library
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from utility import DATA_PATH

""" Load Mall Customer Data """

file_name = 'Mall_Customers.csv'
mall_data = read_csv(DATA_PATH + file_name)

""" Rename Columns """
mall_data = mall_data.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending"})


# Plot Mall Customer Data
mall_data.plot.scatter(x="Income", y="Spending")
plt.show()

# Clustering Data using K-Means:
mall_df = mall_data[["Income", "Spending"]]

k_means = KMeans(n_clusters=5).fit(mall_df)
centroids = k_means.cluster_centers_
print("Centroid of Clustering", centroids)

""" Plot Mall Clusters """
plt.scatter(mall_df['Income'], mall_df['Spending'],
            c=k_means.labels_.astype(float),
            s=50,
            alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)
text = ['c0', 'c1', 'c2', 'c3', 'c4']
for i in range(len(centroids)):
    plt.annotate(text[i], (centroids[i][0], centroids[i][1] + 4))
plt.title('Mall Costumer!!')
plt.xlabel('income'), plt.ylabel('spending')
plt.show()
