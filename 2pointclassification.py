import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('insurance_data.csv')
print(df.head())
plt.scatter(df.age, df.bought_insurance, marker = '+', color = 'red')
plt.show()
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size = 0.8)
X_test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
X_test
y_pred = model.predict(X_test)
model.predict_proba(X_test)

model.score(X_test, y_test)
y_pred
X_test
model.coef_
model.intercept_

import math
def sigmoid(x):
    return 1/(1+math.exp(-x))

def pred_func(age):
    z = 0.042 * age - 1.53
    y = sigmoid(z)
    return y

age = 35
pred_func(age)
age = 43
pred_func(43)