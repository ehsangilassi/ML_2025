from pandas import read_csv
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from numpy import arange
from utility import DATA_PATH
file_name = 'Housing.csv'
data = read_csv(DATA_PATH + file_name)

data.describe()


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
###########################################
""" Lasso Regression with Cross-Validation and Grid Search """

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Lasso(0.02)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

""" Plot Coefficients"""
b = [abs(i) for i in model.coef_]
a = arange(1, 14)
import matplotlib.pyplot as plt
plt.scatter(a, b)
plt.show()

