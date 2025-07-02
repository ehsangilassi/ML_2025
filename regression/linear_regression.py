from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

from utility import DATA_PATH
file_name = 'salary_data.txt'
data = read_csv(DATA_PATH + file_name)

data.describe()

X = data.iloc[:,:-1].values  # independent ---> Array
y = data.iloc[:,1].values  # dependent ---> Vector
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.scatter(x=X, y=y)
plt.show()

""" Separation """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

""" Regression """
regressor = LinearRegression()
regressor.fit(X_train, y_train)

""" Predict """
y_pred = regressor.predict(X_test)

""" Regression Element """
""" y = m*x + intercept """
slope = regressor.coef_
print(f'm: {slope}')
intercept = regressor.intercept_
print(f'Intercept: {intercept}')
regressor.predict([[2.5]])

""" Evaluate"""
# MAE
mean_absolute_error(y_test, y_pred)
# MSE
mean_squared_error(y_test, y_pred)
# RMSE
root_mean_squared_error(y_test, y_pred)
#########################################
# R-squared (coefficient of determination)
print("R^2 Coefficient : ", regressor.score(X_test, y_test))
mean_absolute_percentage_error(y_test, y_pred)

""" Plot """
# Plot Test
plt.scatter(X_train, y_train, color='red')
plt.plot(X_test, y_pred, color='blue')  # plotting the regression line
plt.title("Salary vs Experience (Testing set)")
plt.xlabel("Years of experience")
plt.ylabel("Salaries")
plt.show()