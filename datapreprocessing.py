import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

Titanic = pd.read_csv("titanic.csv")
print(Titanic.columns)
print(Titanic.head())
print(Titanic.shape)
print(Titanic.isnull().sum())
sns.heatmap(Titanic.isnull(), cmap = "spring")
plt.show()
print(Titanic.head())
Titanic.drop("Deck", axis = 1, inplace = True)
print(Titanic.head())
Titanic.dropna(inplace = True)
sns.heatmap(Titanic.isnull(), cbar = False)
plt.show()
print(Titanic.isnull().sum())
sex = pd.get_dummies(Titanic["Sex"], drop_first = True, dtype = int).head(4)
pd.get_dummies(Titanic["Embarked"]).head(4)
arked = pd.get_dummies(Titanic["Embarked"],drop_first = True, dtype = int).head(4)
Pclass = pd.get_dummies(Titanic["Pclass"], drop_first = True, dtype = int).head(4)
titanic = pd.concat([Titanic,sex,Pclass], axis = 1)
print(titanic.head())