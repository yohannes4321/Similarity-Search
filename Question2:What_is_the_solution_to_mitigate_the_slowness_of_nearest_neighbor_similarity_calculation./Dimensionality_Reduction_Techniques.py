"""
The curse of dimensionality  
Distances become less informative: In high dimensions, the difference between the distances of data points 
tends to become smaller, making it harder to distinguish between near and far points.
Increased computational cost: Searching through a high-dimensional space becomes computationally expensive 
since each point has more dimensions to compare.
To mitigate this slowness and make nearest neighbor similarity calculations more efficient, we can employ
 the following strategies:

1. Dimensionality Reduction
Reducing the dimensionality of the dataset is one of the most effective solutions to the curse of dimensionality. By transforming high-dimensional data into a lower-dimensional space, you can speed up similarity calculations and reduce the computational burden. There are several methods to achieve this:

a) Principal Component Analysis (PCA)
PCA is a linear method that finds the directions (principal components) along which the variance of the data is maximized. It projects the data into a lower-dimensional space while preserving as much of the variance (and thus the structure) as possible.

Benefits of PCA:

Reduces Noise: By focusing on the most significant components (those with the highest variance), PCA reduces the impact of noise in the data.
Speeds Up Computation: Reducing the number of dimensions makes the nearest neighbor search faster, as there are fewer features to compare.
How it works:

PCA computes the eigenvectors of the covariance matrix of the data, and the eigenvalues tell us how much variance is captured by each component.
You can then project the data into the lower-dimensional space spanned by the top k eigenvectors.
Code Example (PCA):
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
from sklearn.neighbors import NearestNeighbors

# Generate random data (1000 points, each with 100 dimensions)
data = np.random.rand(1000, 100)

# Apply PCA to reduce the dimensionality to 10
pca = PCA(n_components=10)
reduced_data = pca.fit_transform(data)

# Perform nearest neighbor search in the reduced space
query = np.random.rand(1, 100)  # Random query point
reduced_query = pca.transform(query)  # Project query into PCA space

# Nearest neighbor search
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(reduced_data)
distances, indices = neighbors.kneighbors(reduced_query)

print("Nearest neighbors indices:", indices)
print("Distances:", distances)


# Generate random data (1000 points, each with 100 dimensions)
data = np.random.rand(1000, 100)

# Apply UMAP to reduce the dimensionality to 10
umap_model = umap.UMAP(n_components=10)
reduced_data = umap_model.fit_transform(data)

# Perform nearest neighbor search in the reduced space
query = np.random.rand(1, 100)  # Random query point
reduced_query = umap_model.transform(query)  # Project query into UMAP space

# Nearest neighbor search
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(reduced_data)
distances, indices = neighbors.kneighbors(reduced_query)

print("Nearest neighbors indices:", indices)
print("Distances:", distances)
"""
2. Approximate Nearest Neighbor (ANN) Search Algorithms
Instead of performing an exact nearest neighbor search, Approximate Nearest Neighbor (ANN) algorithms can speed up 
the process by sacrificing a 
little accuracy for performance. Some of the most popular ANN algorithms are:

HNSW (Hierarchical Navigable Small World Graphs): A graph-based index that allows fast approximate nearest neighbor search by
 navigating through a hierarchical structure of connected points.
Locality Sensitive Hashing (LSH): A technique that hashes data points in such a way that similar points are more likely to be mapped 
to the same hash bucket, allowing for quick filtering of candidates.
These methods are effective for large, high-dimensional datasets and can be combined with dimensionality 
reduction techniques like PCA for better performance.

"""
import numpy as np
import faiss

# Generate random high-dimensional data (1000 points, each with 128 dimensions)
data = np.random.random((1000, 128)).astype('float32')

# Build HNSW index
index = faiss.IndexHNSWFlat(128, 32)  # 128 is the dimension, 32 is the number of neighbors to connect to
index.add(data)  # Add data points to the index

# Query point
query = np.random.random((1, 128)).astype('float32')

# Perform the search for the top 5 nearest neighbors
k = 5
distances, indices = index.search(query, k)

print("Nearest neighbors indices:", indices)
print("Distances:", distances)
"""
3. Hybrid Approaches
Combining dimensionality reduction and ANN methods can provide even more significant speedups. For example:

First, reduce the dimensions using PCA or UMAP to make the data more compact.
Then, apply HNSW or LSH for fast nearest neighbor search in the reduced space.
By using hybrid approaches, you can leverage the strengths of both techniques—dimensionality 
reduction to simplify the data and approximate methods to speed up the search.
"""