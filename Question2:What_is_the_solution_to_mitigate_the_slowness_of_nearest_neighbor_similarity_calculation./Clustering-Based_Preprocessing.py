""" K-Means Clustering and Hierarchical Clustering for Nearest Neighbor Search
Clustering is an essential technique used to partition large datasets into smaller, more manageable groups.
Both K-Means and Hierarchical Clustering are popular methods for grouping data points based on similarity.
When applied to Nearest Neighbor (NN) Search, these clustering techniques can reduce the computational burden by narrowing down the search space.  
K-Means Clustering for Nearest Neighbor Search
K-Means Clustering is a partitioning-based clustering method that divides data points into a predefined 
number of clusters, k. Each cluster is represented by its centroid, which is the mean of all the points in that cluster.


K-Means works by iteratively partitioning the data into k clusters. It does this by assigning each point to the nearest centroid and then recalculating the centroids until they stabilize.
The centroids are the "average" position of all the points within the cluster."""

 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Generate random high-dimensional data (e.g., 1000 data points, each with 128 features)
data = np.random.random((1000, 128))

# Step 1: Apply K-Means clustering
kmeans = KMeans(n_clusters=10)  # Number of clusters (k)
kmeans.fit(data)

# Step 2: Assign query point to nearest centroid
query = np.random.random((1, 128))  # Random query point
query_cluster = kmeans.predict(query)

# Step 3: Perform nearest neighbor search within the identified cluster
# Identify points within the cluster
cluster_points = data[kmeans.labels_ == query_cluster]

# Step 4: Find the nearest neighbors in the selected cluster
neighbors = NearestNeighbors(n_neighbors=5)  # Number of nearest neighbors
neighbors.fit(cluster_points)
distances, indices = neighbors.kneighbors(query)

print("Nearest neighbors:", indices)
print("Distances:", distances)
"""
Explanation of Code:
Step 1: We use K-Means from sklearn to partition the data into k=10 clusters.
Step 2: We predict which cluster the query point belongs to by finding the nearest centroid.
Step 3: We select the data points that belong to the identified cluster.
Step 4: We perform a nearest neighbor search within the relevant cluster using NearestNeighbors to find the top 5 nearest neighbors.
Hierarchical Clustering for Nearest Neighbor Search
Hierarchical Clustering builds a tree-like structure called a dendrogram that represents nested clusters of data points. There are two main types:

Agglomerative: Starts with each point as its own cluster and iteratively merges the closest pairs of clusters.
Divisive: Starts with all points in one cluster and recursively splits it into smaller clusters.
How Hierarchical Clustering Works for NN Search:
Tree Structure:
The result of hierarchical clustering is a tree (or dendrogram), where each node represents a cluster. The leaf nodes represent individual data points, and the non-leaf nodes represent groups of data points.
NN Search with Hierarchical Clustering:
During a nearest neighbor search, you can traverse the tree to identify the branch where the query point is most likely to belong.
Once you identify the relevant branch, you can narrow down the search to that branch or its neighboring branches, reducing the search space.
Unlike K-Means, hierarchical clustering doesn't require you to specify the number of clusters upfront, providing more flexibility.
Benefits of Hierarchical Clustering for NN Search:
Tree-Based Search: The tree structure allows for efficient search. You can quickly move down the tree to the most relevant clusters.
No Need to Predefine Number of Clusters: Unlike K-Means, hierarchical clustering doesn't require you to decide on the number of clusters beforehand. You can cut the tree at different levels to obtain varying granularities of clusters.
Focus on Relevant Branches: Once you identify the relevant branch, you can limit your search to just that branch or the closest ones.
Hierarchical Clustering Code Example:
 """
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

# Generate random high-dimensional data (e.g., 1000 data points, each with 128 features)
data = np.random.random((1000, 128))

# Step 1: Apply Agglomerative Clustering (Hierarchical Clustering)
agg_clust = AgglomerativeClustering(n_clusters=10)  # Number of clusters
agg_clust.fit(data)

# Step 2: Assign query point to nearest cluster
query = np.random.random((1, 128))  # Random query point
query_cluster = agg_clust.fit_predict(query)

# Step 3: Perform nearest neighbor search within the identified cluster
# Identify points within the cluster
cluster_points = data[agg_clust.labels_ == query_cluster]

# Step 4: Find the nearest neighbors in the selected cluster
neighbors = NearestNeighbors(n_neighbors=5)  # Number of nearest neighbors
neighbors.fit(cluster_points)
distances, indices = neighbors.kneighbors(query)

print("Nearest neighbors:", indices)
print("Distances:", distances)
