"""why Nearset neighbor similarity is slow

arrises from the neeed to compute the distance between a query point 
and every other point in data set .

the compitotional cost become very slow as data set grows

main reason for slowness
1 high dimensional spaces 

    Euclidean lose meaning as distance between meaning is less meaningfull.
    increased computional cost with out meaningfull result 
2 Large Data set Size
    compute distance for all points requires O(N) operation per query
    for M queries O(M.N) which is inefficient for large data set 
3 Brute force search 
all pairwise distance calculated and sorted and k nearset neighbours are selected 


Techinques to Speed Up Nearest Neighbor Search


1 Approximate Nearest Neighbor (ANN)

avoid exhaustive pairwaise comparisons by stratigically  indexing the data 
and making approximations

exact accuracy for massive performance gains for achieving result close 


Principles
  1 data partitioning :Divide the dataset into smaller and more mangeable chunks
2 Efficent Indexing use tree based ,hash based and graph based structures to quickly identify
approximate neighbors
3 Approximation Allow a small margin of error to speed up the competations

Popular ANN Algorithms and Techniques 

a Faiss (Facebook Ai similarity Search )


    1 optimized Cpu and GPU based implementations

        -Efficent Matrix Operation which have libraries like Blass (Basic linear Algebra )
For fast matrix 
        - Single instruction multiple data Instructions process multiple data point 
in parrallel to speed up calculation

          -GPU Accelaration 
    2 High performance libraries like HNSW (Hierarchical Navigable Small World)

B Vector Quantization

vector quantization is a technique used to compress high dimiensional vectors into smaller repersentaion 

whicn makes faster and less memory 

high-dimensional  data is mapped to smaller set or representative data
instead of comparing the query to every vector in the data set comparsions are
made to smaller set of centroid

IVF inverted File Index 
The core technique used in Faiss to narrow down the search space 
how does it work :
    1 ,Clustring with k means
     Faiss divides to culuster using k means clustring algorthim 

     2 Query Assignment 
     when query is processed it matched centroid of few relevant clusters insted of 
     entire data set
     3 searching within Cluster  """



# using brute force method  IndexFlatL2
import faiss
import numpy as np

 
num_vectors = 10000
dim = 128
dataset = np.random.random((num_vectors, dim)).astype('float32')
query = np.random.random((5, dim)).astype('float32')
 
index = faiss.IndexFlatL2(dim)
index.add(dataset)

 
k = 3
distances, indices = index.search(query, k)

print("Nearest Neighbors (Indices):", indices)
print("Nearest Neighbors (Distances):", distances)

# using IVF 

 
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(dim)  # Base index
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)

 
index_ivf.train(dataset)
index_ivf.add(dataset)  

# Perform search
distances, indices = index_ivf.search(query, k)

print("Nearest Neighbors (Indices):", indices)
print("Nearest Neighbors (Distances):", distances)

"""
Annoy library developed spoitfy for efficient approximate nearest neighbour

A  Tree Based Approch 
    how does it works 
    1 Random Projections Annoy reandomly selects Hyperplanes to split the dataset 
    the split are stored in tree data structures 

    2 Tree Nodes Each node in the tree represents a subset of vectors narrowing 
    down the search space for queries 

    3 Leaves :
    Leaf node contain closest vectors to be compared with query 

    when i get the query i need to not to compare the whole vectors but relevant nodes

B Multiple Tress for Accuracy 
    Annoy builds multiple tress with slightly diffrent random splitting stratagy 

    when A query is made

    the query vector traverse multiple tress
    result form all tree combined to improve accuracy 

C Disc Friendly 
Annoy is highly efficent for large dataset that can not fit into memory 


Trees are stored in memory-mapped files, allowing Annoy to efficiently query data directly from the disk.

"""
from annoy import AnnoyIndex
import numpy as np

# Create a dataset: 10000 vectors with 128 dimensions
num_vectors = 10000
dim = 128
dataset = np.random.random((num_vectors, dim)).astype('float32')

# Build Annoy Index
annoy_index = AnnoyIndex(dim, 'euclidean')  # Use Euclidean distance
for i, vector in enumerate(dataset):
    annoy_index.add_item(i, vector)

# Build the index with 10 trees
annoy_index.build(n_trees=10)

# Save the index to disk
annoy_index.save('annoy_index.ann')
# Load the index from disk
annoy_index = AnnoyIndex(dim, 'euclidean')
annoy_index.load('annoy_index.ann')

# Query a random vector
query_vector = np.random.random(dim).astype('float32')
k = 5  # Find 5 nearest neighbors
indices = annoy_index.get_nns_by_vector(query_vector, k, include_distances=True)

print("Nearest Neighbors (Indices):", indices[0])
print("Nearest Neighbors (Distances):", indices[1])
"""


Dimensionality Reduction 

Reduce the number of dimensions while preserving the distance between 
data points to speed up computations

        Principal Components Analysis (PCA)

            linear techinques that projects data into lower dimensional space 
            using directons of maxumum variance 

            reduce time
"""
from sklearn.decomposition import PCA
import numpy as np

# Generate high-dimensional data
data = np.random.random((10000, 512)).astype('float32')  # 10k vectors, 512 dimensions

# Apply PCA
pca = PCA(n_components=128)  # Reduce to 128 dimensions
reduced_data = pca.fit_transform(data)

print("Original Shape:", data.shape)
print("Reduced Shape:", reduced_data.shape)

"""
Autoencoders 

Neural networks designed to compress and reconstract data 
can handle non linear relationship sutable for complex data types 

"""
import tensorflow as tf
from tensorflow.keras import layers

# Autoencoder model
input_dim = 512
encoding_dim = 128

input_data = tf.keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_data)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(input_data, decoded)
encoder = tf.keras.Model(input_data, encoded)

# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
data = np.random.random((10000, 512)).astype('float32')
autoencoder.fit(data, data, epochs=10, batch_size=256)

reduced_data = encoder.predict(data)
print("Reduced Data Shape:", reduced_data.shape)


"""
Tree Based indexing 
    KD Tree k dimensional Tree
        Efficently divide space into nested rectangle 

        split data recursively along one dimension at each level
        each split is rougly equal partitioning result binary tree structure 
        good for smaller number dimensions 
        Adavantages
        Efficnent with smaller number of dimension 
        simple to implement 

        Disadvantage 
        ineffient in high dimensional spaces due to curse of dimensionality 


    Ball-Tree
    partition data based on proximity not alignment 
    works better for non euclidean distance 
    Groups data into clusters using hyperspheres instead 



"""

from sklearn.neighbors import BallTree
import numpy as np

 
data = np.random.random((10000, 128)).astype('float32')   
query = np.random.random((1, 128)).astype('float32')

 
tree = BallTree(data)
distances, indices = tree.query(query, k=3)

print("nearest neighbors Indices:", indices)
print("nearest neighbors Distances:", distances)

"""
Quantization Techiniques

Qunatization compress data to reduce memory usage and speeds up siimilarity 

    Product Quantization (pq)
    Compress high-dimensional data into smaller subspace while preserving similarity 
    steps
    splits each vector into multiple sub-vectors 
    quantize (Compress each subvector indpendently )
     advantages
     reduce storage requirements 
     very effiencent high dimensional data 

     Disadvantage 
     slight loss of accuracy due to approximation 


"""
import faiss
import numpy as np

# Generate data
data = np.random.random((10000, 128)).astype('float32')

# Product Quantization Index
quantizer = faiss.IndexFlatL2(128)
pq_index = faiss.IndexIVFPQ(quantizer, 128, 100, 8, 8)   
pq_index.train(data)
pq_index.add(data)

# Query
query = np.random.random((1, 128)).astype('float32')
distances, indices = pq_index.search(query, k=3)

print("Nearest Neighbors Indices:", indices)
print("Nearest Neighbors Distances:", distances)
