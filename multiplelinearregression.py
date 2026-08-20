import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"petrol_consumption.csv")
print(dataset.head())
print(dataset.describe())

X = dataset[["Petrol_tax", "Average_income", "Paved_Highways", "Population_Driver_licence(%)"]]
y = dataset["Petrol_Consumption"]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
coefficiant_df = pd.DataFrame(reg.coef_, X.columns, columns = ['Coefficiant'])
print(coefficiant_df)

y_prediction = reg.predict(X_test)
df = pd.DataFrame({"Actual":y_test,"Predicted":y_prediction})
print(df)

from sklearn import metrics
print("Average absoulute error(MAE):",metrics.mean_absolute_error(y_test,y_prediction))
print("Average squared error(MSE):",metrics.mean_squared_error(y_test,y_prediction))
print("Square root of average squared error(RMSE):",np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))