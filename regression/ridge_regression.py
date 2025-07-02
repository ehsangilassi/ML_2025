import numpy as np
from numpy import mean, absolute, std
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from numpy import arange
from utility import DATA_PATH
import seaborn as sns
import matplotlib.pyplot as plt

file_name = 'Housing.csv'
data = read_csv(DATA_PATH + file_name)
sns.pairplot(data)
plt.show()

data.describe()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Ridge(alpha=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)

#################################
cv = KFold(n_splits=5, random_state=1, shuffle=True)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Model with alpha=1
model = Ridge(alpha=1.0)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f' % (mean(scores)))
np.std(scores)
###########################################
# Grid Search for Hyperparameter Tuning
# define model
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
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