from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from utility import DATA_PATH

# Read Dataset
file_name = 'Social_Network_Ads.txt'
dataset = read_csv(DATA_PATH + file_name)

# Preprocessing Data
#  Encode Sex Column
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset = dataset.drop(['User ID'], axis=1)

#  Separation Data"""
X = dataset.drop(['Purchased'], axis=1).values
y = dataset.iloc[:, -1].values

""" Separation Train and Test Data """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

""" Normalize Data"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

""" Implementation Logistic Regression"""
logreg = LogisticRegression(solver='lbfgs', random_state=1, penalty='l2')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

""" Evaluate"""

classes_names = ['Not Purchased', 'Purchased']
confusion_matrix(y_test, y_pred)
cm = DataFrame(confusion_matrix(y_test, y_pred), index=classes_names, columns=classes_names)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

target_names = ['Not Purchased', 'Purchased']
print(classification_report(y_test, y_pred, target_names=target_names))

acc_score = accuracy_score(y_test, y_pred)

""" Plot ROC Curve """
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# p = dataset[dataset['Purchased'] == 1]
# not_p = dataset[dataset['Purchased'] == 0]
# p.Age.plot(kind='hist')
# plt.show()
# not_p.Age.plot(kind='hist')
# plt.show()
