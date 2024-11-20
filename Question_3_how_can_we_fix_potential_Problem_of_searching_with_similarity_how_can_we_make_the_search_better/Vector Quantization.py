"""What is Vector Quantization?
Vector quantization is a technique used to compress high-dimensional vectors by reducing the number of bits 
required to represent each vector. Instead of storing each vector with its full precision (often using 32-bit
floats or higher), vector quantization maps each vector to a "codebook" of representative vectors, thereby 
reducing the amount of data needed to store or process these vectors.
For example, if you have a vector with 1,536 dimensions, storing it in full precision (using 32-bit float) might take up around 6 KB. 
With a large collection of millions of vectors, this adds up quickly. Vector quantization helps by reducing the memory usage, allowing the vectors to be stored in a compressed form, often in just a fraction of the space.
Key Methods of Vector Quantization
Scalar Quantization: Scalar quantization involves scaling each dimension of a vector into a smaller range. For example, if each dimension of a vector is stored as a 32-bit float, it can be reduced to a smaller integer representation (e.g., int8), which uses 1 byte instead of 4 bytes. This is done by "rounding" the values into a smaller fixed range. 
This reduces the memory consumption and speeds up both storage and processing.
Example: If your data is in the range from -1.0 to 1.0, scalar quantization can map this range into integers between -128 and 127 (the range of an 8-bit integer). This reduces memory usage by 75%.

Product Quantization: In product quantization, the vector is split into multiple sub-vectors, and each sub-vector is quantized independently. This method is more efficient than scalar quantization, especially for high-dimensional vectors.
Optimized Search with HNSW Index: HNSW (Hierarchical Navigable Small World) is a state-of-the-art indexing structure used to perform fast approximate nearest neighbor (ANN) searches in high-dimensional spaces. It builds a multi-layer graph where each node is a vector, and the edges between them represent "shortcuts" to nearby vectors. This allows you to quickly jump through the graph and find the nearest neighbors without searching the entire dataset.
 
 
"""
 

 
import faiss
import numpy as np

# Generate random high-dimensional data
dimension = 128  # Dimensionality of the vectors
num_vectors = 10000  # Number of vectors

# Create random vectors
data = np.random.random((num_vectors, dimension)).astype('float32')

# Step 1: Vector quantization using scalar quantization (8-bit)
# Create a quantizer
quantizer = faiss.IndexFlatL2(dimension)

# Compress the vectors using scalar quantization
nlist = 100  # Number of clusters for quantization
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
index.train(data)  # Train the index with data

# Step 2: Create HNSW index for fast approximate search
# Use HNSW index on the quantized data
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors
hnsw_index.add(data)

# Step 3: Perform a search
query = np.random.random((1, dimension)).astype('float32')  # Random query vector

# Perform the search
k = 5  # Number of nearest neighbors to find
D, I = hnsw_index.search(query, k)

print("Distances: ", D)
print("Indices of nearest neighbors: ", I)
"""
Data Generation: We generate random high-dimensional data to simulate vectors in a real-world dataset.

Scalar Quantization:

We use IndexIVFFlat from FAISS to apply vector quantization (IVF: Inverted File). This is one of the quantization methods supported by FAISS.
HNSW Index:

The IndexHNSWFlat creates an HNSW index, which organizes the data in a graph structure. 
The parameter 32 represents the number of neighbors to consider when building the graph.
Search:

We create a random query vector and search for the 5 nearest neighbors in the quantized and indexed dataset using HNSW. The result includes
 the distances and indices of the nearest neighbors.
How Can We Improve Search with Similarity?
To improve search accuracy and efficiency in large datasets, consider the following:

Hybrid Approaches: Combine vector quantization with advanced indexing structures like HNSW, IVF, and PQ (Product Quantization) 
to balance memory efficiency and search accuracy.

Tuning Parameters: Experiment with different quantization methods, such as scalar vs. product quantization, and optimize HNSW parameters (e.g., the number of neighbors in the graph) to improve speed and accuracy.

Fine-Tuning the Index: For highly dynamic data (frequent updates), consider using more flexible indexing
 techniques that allow for fast insertion of new vectors without requiring the entire index to be rebuilt."""