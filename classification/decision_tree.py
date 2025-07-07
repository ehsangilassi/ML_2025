from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from utility import DATA_PATH
""" Read Data"""
""" 
Data Set :
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, 
based on certain diagnostic measurements included in the dataset. Several constraints were placed on 
the selection of these instances from a larger database. In particular, all patients here are females 
at least 21 years old of Pima Indian heritage.

"""
file_name = "pima-indians-diabetes.csv"
data_set = read_csv(DATA_PATH + file_name)
data_set.head()
data_set.describe()
X = data_set.drop('Class', axis=1)
y = data_set['Class']

""" Separate test and train"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

""" Classification """
classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)

classifier.fit(X_train, y_train)

""" Prediction """
y_pred = classifier.predict(X_test)

""" Evaluation """
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print(" Classification Report : ")
print(classification_report(y_test, y_pred))

print(" Accuracy : ", accuracy_score(y_test, y_pred))
