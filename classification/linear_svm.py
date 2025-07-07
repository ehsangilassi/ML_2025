from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from utility import DATA_PATH

""" Read Data"""

file_name = 'BankNote_Authentication.csv'
""" Bank Note Authentication Dataset:
About this file

Data were extracted from images that were taken from genuine and 
forged banknote-like specimens. For digitization, an 
industrial camera usually used for print inspection was used. 
The final images have 400x 400 pixels. Due to the object lens 
and distance to the investigated object gray-scale 
pictures with a resolution of about 660 dpi were gained. 
Wavelet Transform tool were used to extract features from images.
"""

bank_data_set = read_csv(DATA_PATH + file_name)
bank_data_set.head()

X = bank_data_set.drop('class', axis=1)
y = bank_data_set['class']

""" Separate test and train"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)

""" Classification """
sv_classifier = SVC(kernel='linear')
sv_classifier.fit(X_train, y_train)

""" Predict test data """
y_pred = sv_classifier.predict(X_test)

""" Confusion Matrix"""
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


