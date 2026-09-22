import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Read the dataset
df = pd.read_csv("customers.csv")

# 2. Display first 5 rows
print(df.head())

# 3. Select columns for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Find inertia for different numbers of clusters
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 5. Draw Elbow Graph
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 6. Create 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# 7. Predict the cluster for each customer
df['Cluster'] = kmeans.fit_predict(X)

# 8. Display the result
print("\nCustomer data with clusters:")
print(df.head(20))

# 9. Visualize Age vs Spending Score
plt.scatter(
    df['Age'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    s=100
)

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Clusters")
plt.show()