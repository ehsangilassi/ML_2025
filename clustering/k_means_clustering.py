from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

""" Read Data """
path = 'D:/z_during/machineLearning/data/'
file_name = 'data_checkin_loc.csv'
df = read_csv(path + file_name)
print(df)

""" Plot Check_in Data for Tehran """
df.plot.scatter(x='X', y='Y', s=20)
plt.title('Check-in Data in Tehran')
plt.xlabel('Longitude'), plt.ylabel('Latitude')
plt.show()

""" Use K-Means for Clustering """
k_means = KMeans(n_clusters=3).fit(df)
centroids = k_means.cluster_centers_
print("Centroid of Clustering", centroids)
# df["label"] = label_kmean

""" Plot Clustering """
plt.scatter(df['X'], df['Y'], c=k_means.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, label=[0, 1, 2])
text = ['c0', 'c1', 'c2']
for i in range(len(centroids)):
    plt.annotate(text[i], (centroids[i][0], centroids[i][1] + 4))
plt.title('Check in data is cluster!!')
plt.xlabel('X'), plt.ylabel('Y')
plt.show()
""" Add a test data to Predict Cluster """
sample_coordinate = [538000, 3948000]
df_sample = DataFrame({"X": [sample_coordinate[0]], "Y": [sample_coordinate[1]]})
predict_cluster = k_means.predict(df_sample)
print("predict_cluster : ", predict_cluster)
plt.scatter(sample_coordinate[0], sample_coordinate[1], cmap='spring')
plt.text(sample_coordinate[0], sample_coordinate[1], "Test Data")
plt.show()


""" Evaluation Clustering """

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

model = KElbowVisualizer(KMeans(), k=10)
model.fit(df)
model.show()
print("Silhouette each k: ", model.k_scores_)
print("Time fitting for each k: ", model.k_timers_)
print("The optimal value of k: ", model.elbow_value_)
print("The silhouette optimal value of k:", model.elbow_score_)
