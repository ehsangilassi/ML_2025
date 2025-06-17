from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

from utility import DATA_PATH

""" Read Data """
file_name = 'unbalanced.txt'
df = read_csv(DATA_PATH + file_name, sep=' ', names=["X", "Y"])
print(df)

""" Plot Check_in Data for Tehran """
df.plot.scatter(x='X', y='Y', s=20)
plt.title('Unbalanced Data')
plt.xlabel('Longitude'), plt.ylabel('Latitude')
plt.show()

""" Use Mean-Shift for Clustering """
mean_shift = MeanShift(bandwidth=25000)
mean_shift.fit(df)
mean_shift_centers = mean_shift.cluster_centers_
print("Centroid of Clustering", mean_shift_centers)

""" Plot Clustering """
plt.scatter(df['X'], df['Y'], c=mean_shift.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(mean_shift_centers[:, 0], mean_shift_centers[:, 1], c='red', s=100)
plt.title('Mean-Shift Clusters of unbalanced')
plt.xlabel('Longitude'), plt.ylabel('Latitude')
plt.show()
