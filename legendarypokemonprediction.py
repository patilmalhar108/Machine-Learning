import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("pokemondata.csv")
print(data.head())
data['Type 2'].fillna(value = 'none', inplace = True)
print(data.isnull().sum())

data['Type 1'].value_counts().plot.bar()
plt.show()
data['Type 2'].value_counts().plot.bar()
plt.show()
data['Legendary'].value_counts().plot.bar()
plt.show()

print(data['Type 1'].unique())
print(data['Type 2'].unique())

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data['Legendary'] = lb.fit_transform(data['Legendary'])
print(data.head())

data.drop('Name', axis = 1, inplace = True)
data = pd.get_dummies(data)
print(data.shape)
x = data.drop('Legendary', axis = 1)
y = data['Legendary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
LogReg = LogisticRegression(max_iter = 1000)
LogisticRegression.fit(X_train, y_train)
ypred1 = LogReg.predict(X_test)
accuracy_score(y_test, ypred1)
from sklearn.neighbors import KNeighborsClassifier
errorrate = []
for a in range(1,40):
    k = a
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    errorrate.append(np.mean(y_test - preds))

plt.figure(figsize = (10,7))
plt.plot(range(1,40),errorrate, color = 'blue', linestyle = 'dashed', marker = '0',
markerfacecolor = 'red', markersize = 10)
plt.title("Errorrate vs K Value")
plt.xlabel("K")
plt.ylabel("Errorrate")
plt.show()
knn_model = KNeighborsClassifier(n_neighbors = 8)
knn_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
ypred = knn_model.predict(X_test)
accuracy_score(y_test, ypred)
from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier()
clf_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
ypred = clf_model.predict(X_test)
accuracy_score(y_test, ypred)