from sklearn.cluster import MiniBatchKMeans
from pandas import read_csv
import matplotlib.pyplot as plt

from utility import DATA_PATH


file_name = 'data_checkin_loc.csv'
df = read_csv(DATA_PATH + file_name)
print(df)

""" Plot Check_in Data for Tehran """
df.plot.scatter(x='X', y='Y', s=20)
plt.title('Check-in Data in Tehran')
plt.xlabel('Longitude'), plt.ylabel('Latitude')
plt.show()

""" MiniBatchKMeans Algorithm """
mbk_means = MiniBatchKMeans(n_clusters=3, max_iter=200, batch_size=200)
mbk_means.fit(df)
mbk_centers = mbk_means.cluster_centers_
print(mbk_centers)
