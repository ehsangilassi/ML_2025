from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

from utility import DATA_PATH
# Importing the dataset

data = read_csv('https://github.com/ybifoundation/Dataset/raw/main/Economy%20of%20Scale.csv')

sns.pairplot(data)
plt.show()

""" Linear Regression """
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# """ Separation """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# """ Predict """
y_pred = regressor.predict(X_test)

# """ Score """
r2 = r2_score(y_test, y_pred)
print("R-squared in Linear Regression: ", r2)
mae_percentage = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error in Linear Regression: ", mae_percentage)

# Plot
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='red')
ax.scatter(X_train, regressor.predict(X_train), color='blue')
ax.set_ylabel('Cost Per Unit Sold [dollars]')
ax.set_xlabel('Number of Units [in Millions]')
ax.set_title('Unit Cost vs. Number of Units [in Millions](Training dataset)')
plt.show()


""" Polynomial Regression """
poly_reg = PolynomialFeatures(degree=4)
# increase Degree to 4
X_poly = poly_reg.fit_transform(X)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y,  test_size=0.20, random_state=0)

reg_poly = LinearRegression()
reg_poly.fit(Xp_train, yp_train)
print(f'poly_intercept: {reg_poly.intercept_}')
print(f'poly_coefficients: {reg_poly.coef_}')

""" Predict """
yp_pred = reg_poly.predict(Xp_test)
mean_absolute_percentage_error(yp_test, yp_pred)
r2_poly = r2_score(yp_test, yp_pred)
print("R-squared in Polynomial Regression: ", r2_poly)
mae_percentage_poly = mean_absolute_percentage_error(yp_test, yp_pred)
print("Mean Absolute Percentage Error in Polynomial Regression: ", mae_percentage_poly)

""" plot Poly Result """
fig,ax = plt.subplots()
ax.scatter(X_train, y_train, color='red')
ax.scatter(X_train, reg_poly.predict(Xp_train), color='blue')
ax.set_ylabel('Cost Per Unit Sold [dollars]')
ax.set_xlabel('Number of Units [in Millions]')
ax.set_title('Unit Cost vs. Number of Units [in Millions](Training dataset)')
plt.show()
