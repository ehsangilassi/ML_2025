from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from utility import DATA_PATH


file_name = 'iris.csv'
data = read_csv(DATA_PATH + file_name)

data.describe()

X = data.drop('variety', axis=1)
y = data["variety"]

""" Encode iris type """
le = LabelEncoder()
y = le.fit_transform(y)

""" Separation """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

""" Normalize Data"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


""" Implementation SVM Classification with POLY kernel"""
from numpy import arange
from datetime import datetime
degreeG = list(arange(2, 6))
gammaG = list(arange(0, 1.1, 0.1))
cSetG = list(arange(0.1, 1.1, 0.1))


t1 = datetime.now()
from sklearn.model_selection import GridSearchCV
model = SVC(kernel='poly')
gridSearch = GridSearchCV(estimator=model,param_grid={'degree':degreeG,'C':cSetG,'gamma':gammaG},cv=4)
gridSearch.fit(X,y)


print(f'The best parameter is {gridSearch.best_params_}')
print(f'The best score is {gridSearch.best_score_}')
print(f'The best estimator is {gridSearch.best_estimator_}')
t2 = datetime.now()
print(f'The time taken is {t2-t1}')

sv_poly_classifier = SVC(kernel='poly', degree=2, C=0.1, gamma=0.8)
sv_poly_classifier.fit(X_train, y_train)

""" Prediction """
y_pred_poly = sv_poly_classifier.predict(X_test)

""" Evaluate"""
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_poly))
print("Confusion Matrix With Parameter:")
print(classification_report(y_test, y_pred_poly))
"""
Experiment Data:
----------------------------------
DEGREE  |   C   |   ACCURACY     |
----------------------------------
    8   |  0.01 |     0.40
----------------------------------
    8   |   2   |     0.83
----------------------------------
    8   |   10  |     0.80
----------------------------------
    5   |   0.1 |     0.87
----------------------------------
    10  |   1   |     0.83
----------------------------------
    3   |   1   |     0.90
----------------------------------
"""


"#########################################################################"
""" Implementation SVM Classification with rbf kernel"""
sv_rbf_classifier = SVC(kernel='rbf', gamma='scale', C=1)
sv_rbf_classifier.fit(X_train, y_train)

""" Prediction """
y_pred_rbf = sv_rbf_classifier.predict(X_test)

""" Evaluate"""
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))
print("Confusion Matrix With Parameter:")
print(classification_report(y_test, y_pred_rbf))

"###############################################################"
""" Implementation SVM Classification with Sigmoid kernel"""
sv_sigmoid_classifier = SVC(kernel='sigmoid')
sv_sigmoid_classifier.fit(X_train, y_train)

""" Prediction """
y_pred_sigmoid = sv_sigmoid_classifier.predict(X_test)

""" Evaluate"""
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_sigmoid))
print("Confusion Matrix With Parameter:")
print(classification_report(y_test, y_pred_sigmoid))

"""
Accuracy table of kernel:
----------------------------------
      kernel     |   ACCURACY     |
----------------------------------
    Polynomial   |      0.90      |
----------------------------------
        RBF      |       1        |
----------------------------------  
    sigmoid      |       0.87
    
"""

""" Hyperparameter Tuning """
file_name = 'iris.csv'
data = read_csv(DATA_PATH + file_name)

data.describe()

X = data.drop('variety', axis=1)
y = data["variety"]

""" Encode iris type """
le = LabelEncoder()
y = le.fit_transform(y)

from numpy import arange
from datetime import datetime
# degreeG = [0, 1, 6]
# cSetG = [0.1, 1, 10]
# gammaG = [0.1, 1, 10]
degreeG = list(arange(1, 10, 2))
gammaG = list(arange(0, 1.1, 0.25))
cSetG = list(arange(0.1, 1.1, 0.1))
kernelsG = ['linear', 'poly', 'rbf']


t1 = datetime.now()
from sklearn.model_selection import GridSearchCV
model = SVC()
gridSearch = GridSearchCV(estimator=model,
                          param_grid={'degree': degreeG,
                                                      'kernel': kernelsG,
                                                      'C': cSetG,
                                                      'gamma':gammaG},cv=4)
gridSearch.fit(X,y)
print(f'The best parameter is {gridSearch.best_params_}')
print(f'The best score is {gridSearch.best_score_}')
print(f'The best estimator is {gridSearch.best_estimator_}')
t2 = datetime.now()
print(f'The time taken is {t2-t1}')
