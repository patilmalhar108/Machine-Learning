# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


# Read the dataset
df = pd.read_csv('customers.csv', index_col=0)

# Display first 5 rows
print(df.head())

# Display column names
print(df.columns)

# Display information about dataset
print(df.info())

# Display statistical information
print(df.describe())

# Check missing values
print(df.isnull().sum())


# ---------------------------------------------------
# Distribution plots
# ---------------------------------------------------

plt.figure(figsize=(15, 6))

n = 0

for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1, 3, n)

    plt.subplots_adjust(wspace=0.5)

    sns.histplot(df[x], bins=15, kde=True)

    plt.title('Distribution of {}'.format(x))

plt.show()


# ---------------------------------------------------
# Pairplot
# ---------------------------------------------------

sns.pairplot(
    df,
    vars=[
        'Spending Score (1-100)',
        'Annual Income (k$)',
        'Age'
    ],
    hue='Gender'
)

plt.show()


# ---------------------------------------------------
# Age vs Spending Score
# ---------------------------------------------------

plt.figure(figsize=(15, 7))

plt.title(
    'Scatter Plot of Age vs Spending Score',
    fontsize=20
)

plt.xlabel('Age')
plt.ylabel('Spending Score')

plt.scatter(
    x='Age',
    y='Spending Score (1-100)',
    data=df,
    s=100
)

plt.show()


# ---------------------------------------------------
# K-Means using Age and Spending Score
# ---------------------------------------------------

X1 = df[
    ['Age', 'Spending Score (1-100)']
].values


# Find inertia for different numbers of clusters

inertia = []

for n in range(1, 15):

    algorithm = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=111
    )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)


# Elbow graph

plt.figure(figsize=(15, 6))

plt.plot(
    np.arange(1, 15),
    inertia,
    'o'
)

plt.plot(
    np.arange(1, 15),
    inertia,
    '-',
    alpha=0.5
)

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.title('Elbow Method')

plt.show()


# Create 4 clusters

algorithm = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=111
)

algorithm.fit(X1)

labels1 = algorithm.labels_

centroids1 = algorithm.cluster_centers_


# Plot the clusters

plt.figure(figsize=(15, 7))

plt.scatter(
    X1[:, 0],
    X1[:, 1],
    c=labels1,
    s=100
)

plt.scatter(
    centroids1[:, 0],
    centroids1[:, 1],
    s=300,
    c='red',
    alpha=0.5
)

plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

plt.title('Customer Clusters: Age vs Spending Score')

plt.show()


# ---------------------------------------------------
# K-Means using Annual Income and Spending Score
# ---------------------------------------------------

X2 = df[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].values


# Find inertia

inertia = []

for n in range(1, 11):

    algorithm = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=111
    )

    algorithm.fit(X2)

    inertia.append(algorithm.inertia_)


# Elbow graph

plt.figure(figsize=(15, 6))

plt.plot(
    np.arange(1, 11),
    inertia,
    'o'
)

plt.plot(
    np.arange(1, 11),
    inertia,
    '-',
    alpha=0.5
)

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.title('Elbow Method')

plt.show()


# Create 5 clusters

algorithm = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=111
)

algorithm.fit(X2)

labels2 = algorithm.labels_

centroids2 = algorithm.cluster_centers_


# Plot the clusters

plt.figure(figsize=(15, 7))

plt.scatter(
    X2[:, 0],
    X2[:, 1],
    c=labels2,
    s=100
)

plt.scatter(
    centroids2[:, 0],
    centroids2[:, 1],
    s=300,
    c='red',
    alpha=0.5
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.title(
    'Customer Clusters: Income vs Spending Score'
)

plt.show()


# ---------------------------------------------------
# K-Means using Age, Income and Spending Score
# ---------------------------------------------------

X3 = df[
    [
        'Age',
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]
].values


# Find inertia

inertia = []

for n in range(1, 11):

    algorithm = KMeans(
        n_clusters=n,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=111
    )

    algorithm.fit(X3)

    inertia.append(algorithm.inertia_)


# Elbow graph

plt.figure(figsize=(15, 6))

plt.plot(
    np.arange(1, 11),
    inertia,
    'o'
)

plt.plot(
    np.arange(1, 11),
    inertia,
    '-',
    alpha=0.5
)

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.title('Elbow Method')

plt.show()


# Create 6 clusters

algorithm = KMeans(
    n_clusters=6,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=111
)

# Predict clusters

y_kmeans = algorithm.fit_predict(X3)

# Add cluster number to dataset

df['Cluster'] = y_kmeans


# Display final dataset

print("\nFinal Customer Data:")
print(df.head())


# ---------------------------------------------------
# 3D visualization
# ---------------------------------------------------

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df['Age'],
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    s=100
)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')

ax.set_title(
    'Customer Clusters using Age, Income and Spending Score'
)

plt.show()


# Display final 5 rows

print(df.tail())