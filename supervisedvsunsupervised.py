import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('USA_Housing.csv', index_col = 0)
print(data.head(10))
print(data.info())
print(data.describe())
sns.distplot(data['Price'])
plt.show()
x = data[['Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
'Area Population']]
y = data['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.4, random_state = 101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_, x.columns, columns = ['coeff'])
print(cdf)

pred = lm.predict(X_test)

plt.figure(figsize = (16,8))
plt.scatter(y_test, pred)
plt.show()
plt.figure(figsize = (16,8))
sns.distplot((y_test - pred), bins = 50)
plt.show()

from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, pred))
print("MSE:", metrics.mean_squared_error(y_test, pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pred)))

from sklearn.datasets import make_blobs
data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8,
random_state = 101)
plt.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(data[0])
kmeans.cluster_centers_
kmeans.labels_
f,(ax1, ax2) = plt.subplots(1,2, sharey = True, fig_size = (10,6))
ax1.scatter(data[0][:,0], data[0][:,1], c = kmeans.labels_, cmap = 'rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')