# Multiple Linear Regression
from pandas import read_csv, get_dummies, concat, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import seaborn as sns

from utility import DATA_PATH
# Importing the dataset
file_name = '50_startups.txt'
data_set = read_csv(DATA_PATH + file_name)

""" Plot data """
sns.pairplot(data_set)
plt.show()

""" Scatter Plot """
sns.scatterplot(x='R&D Spend',
                y='Profit', data=data_set)
plt.show()

""" Independent Variables and dependent variable"""
X = data_set.iloc[:, :-1]
# X = data_set.drop(['Profit', 'Administration'], axis=1)
y = data_set.iloc[:, 4]

# Data Cleaning
states = get_dummies(X['State'], drop_first=True)
# Drop the state coulmn
X = X.drop('State', axis=1)
# concat the dummy variables
X = concat([X, states], axis=1)

"""Convert string data to label :"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label = le.fit_transform(data_set['State'])


""" Separation """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""" Create Model"""
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
mean_absolute_error(y_test, y_pred)
root_mean_squared_error(y_test, y_pred)
mean_absolute_percentage_error(y_test, y_pred)
print("R^2 Coefficient : ", regressor.score(X_test, y_test))
print("R^2 Coefficient : ", r2_score(y_test, y_pred))

p = [165000.20, 136000, 480000.10, 0, 1]
import numpy as np
print(regressor.predict(np.array([p])))
