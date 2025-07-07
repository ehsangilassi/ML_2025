from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.datasets import load_iris
from utility import DATA_PATH

file_name = 'iris.csv'
data = read_csv(DATA_PATH + file_name)

data.describe()

X = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values
# AX = L

""" Encode iris type """
le = LabelEncoder()
y = le.fit_transform(y)

""" Separation """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

""" Normalize Data"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" Implementation Logistic Regression"""
knn = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

""" Evaluate"""
confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
prec_score = precision_score(y_test, y_pred, average='micro')
rec_score = recall_score(y_test, y_pred, average='weighted')

"""find best K"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

parameters = {"n_neighbors": range(1, 20, 2)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
