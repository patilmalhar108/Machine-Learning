import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pokemondata.csv')

print(data.info())

data['Type 2'] = data['Type 2'].fillna('None')

data.drop('Name', axis = 1, inplace = True)

data = pd.get_dummies(data)

y = data['Legendary']
x = data.drop('Legendary', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print("Random forest accuracy:", accuracy)
result = pd.DataFrame({'actual':y_test.values, 'predicted':y_pred})
print(result.head(10))