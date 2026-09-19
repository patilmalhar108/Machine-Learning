import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('water_potability.csv')
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.describe())
print(data.info())
print(data.fillna(data.mean(), inplace = True))
data.Potability.value_counts()
sns.countplot(data['Potability'])
plt.show()

sns.histplot(figsize = (14,14))
plt.show()

sns.displot(data['ph'])
plt.show()

plt.figure(figsize = (13,8))
sns.heatmap(data.corr(), annot = True, cmap = 'terrain')
plt.show()

data.boxplot(figsize = (14,7))
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 101, 
shuffle = True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
dt = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 10, splitter = 'best')
dt.fit(X_train, y_train)

pred = dt.predict(X_test)
print(f'Accuracy Score = \n{accuracy_score(y_test, pred)*100}')
print(f'Confusion Matrix = \n{confusion_matrix(y_test, pred)}')
print(f'Classification Report = \n{classification_report(y_test, pred)}')

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 10)
print(knn.fit(X_train, y_train))

pred = knn.predict(X_test)
print(f'Accuracy Score = \n{accuracy_score(y_test, pred)*100}')
print(f'Confusion Matrix = \n{confusion_matrix(y_test, pred)}')
print(f'Classification Report = \n{classification_report(y_test, pred)}')

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
print(log.fit(X_train, y_train))

pred = log.predict(X_test)
print(f'Accuracy Score = \n{accuracy_score(y_test, pred)*100}')
print(f'Confusion Matrix = \n{confusion_matrix(y_test, pred)}')
print(f'Classification Report = \n{classification_report(y_test, pred)}')
