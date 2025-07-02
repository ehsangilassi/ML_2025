"""
In the below code we practice data manipulation using pandas and numpy library."""


# Data URL :https://www.kaggle.com/datasets/arnavgupta1205/usa-housing-dataset?resource=download

# Library:
from pandas import read_csv
import matplotlib.pyplot as plt

from utility import DATA_PATH

""" Functions: """


def read_csv_file(path, file_name):
    df = read_csv(path + file_name)
    return df


df = read_csv_file(DATA_PATH, "usa_housing_kaggle.csv")
df.Price.plot(kind="hist")
plt.show()

