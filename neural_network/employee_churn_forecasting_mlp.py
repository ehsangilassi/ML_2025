from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from utility import DATA_PATH

file_name = 'HR.csv'
data = read_csv(DATA_PATH + file_name)

data.head(2)

""" Pre-processing"""
encoder = preprocessing.LabelEncoder()
data["dep"] = encoder.fit_transform(list(data["dep"]))
data["salary"] = encoder.fit_transform(list(data["salary"]))

""" Separate Data """
X = data.loc[:, data.columns != "left"]
y = data[["left"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" Create Model """
# 1: (6, 5) ----------> acc: 0.938
# 2: (6, 5, 4, 2) ----> acc: 0.949
clf = MLPClassifier(hidden_layer_sizes=(9, 7, 5, 2), # (6, 5, 4, 2)
                    activation='relu',
                    solver='adam',
                    random_state=5,
                    learning_rate_init=0.001,
                    verbose=True
                    )

# Fit Model
clf.fit(X_train, y_train)
# Predict with model
y_pred = clf.predict(X_test)

# Accuracy Calculation
print("Model Accuracy: ", accuracy_score(y_test, y_pred))
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

""" Tuning """
from sklearn.model_selection import GridSearchCV
clf = MLPClassifier()
param_grid = {
    'hidden_layer_sizes': [(6, 5, 4, 2), (9, 7, 5, 2), (6, 5, 3, 2)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

grid = GridSearchCV(clf, param_grid, n_jobs= -1, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)
