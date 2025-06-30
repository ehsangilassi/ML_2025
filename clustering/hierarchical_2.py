from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from utility import DATA_PATH

""" 
Data:
        which contains information for nine different protein sources and 
        their respective consumption from various countries.
"""
df = read_csv('https://raw.githubusercontent.com/LearnDataSci/glossary/main/data/protein.csv')

""" Hierarchical Clustering """
data_features = df.iloc[:, 1:10]
Z2 = linkage(data_features, method='ward', metric='euclidean')
labelList = list(df['Country'])

""" Show Dendrogram"""
plt.figure(figsize=(13, 12))
dendrogram(
    Z2,
    orientation='right',
    labels=labelList,
    distance_sort='descending',
    show_leaf_counts=False,
    leaf_font_size=16
)
plt.show()

""" Clusters """

from scipy.cluster.hierarchy import fcluster

df['Clusters'] = fcluster(Z2, 2, criterion='maxclust')

df.head()
country1 = df[df["Clusters"] == 1]
country2 = df[df["Clusters"] == 2]

