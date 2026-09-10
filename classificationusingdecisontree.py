import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data = pd.read_csv("sample_data-2.csv")
print(data.head())
from sklearn.model_selection import train_test_split
y = data.pop('TARGET CLASS')
x = data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier
clf_model = DecisionTreeClassifier()
clf_model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_predict = clf_model.predict(x_test)
accuarcy = accuracy_score(y_test, y_predict)
print(accuarcy)