"""
1. Approximate Nearest Neighbor (ANN) Search
ANN Search is a technique used for quickly finding the closest data points in high-dimensional spaces, without the computational 
cost of exact nearest neighbor searches. This technique is especially useful when dealing with large datasets or when speed is a priority over
 exact results.

Why ANN?

As the dimensionality of the data increases, exact nearest neighbor searches become computationally expensive
 due to the curse of dimensionality. ANN methods provide a trade-off between search accuracy and computational efficiency.
ANN sacrifices some accuracy for performance, making it ideal when you need fast responses and can tolerate small inaccuracies.
 

 """
import faiss
import numpy as np

# Example: Creating a Faiss index
d = 128  # dimensionality of the vectors
xb = np.random.random((1000, d)).astype('float32')  # 1000 random vectors
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
index.add(xb)  # Adding vectors to the index

# Querying the nearest neighbors
xq = np.random.random((1, d)).astype('float32')  # Query vector
k = 5  # Number of nearest neighbors
D, I = index.search(xq, k)  # D: distances, I: indices of the neighbors
print(I)
"""
Annoy (Approximate Nearest Neighbors Oh Yeah)

Annoy is another library used for high-dimensional search. It's optimized for memory usage and is widely used in recommendation systems.
Example: You could use Annoy to recommend items based on user behavior by indexing user-item interactions.
from annoy import AnnoyIndex"""

# Example: Building an index with Annoy
f = 40  # Dimensionality of the vectors
t = AnnoyIndex(f, 'angular')  # Using 'angular' distance metric
for i in range(1000):
    t.add_item(i, np.random.random(f).tolist())
t.build(10)  # Build 10 trees for approximation

# Querying nearest neighbors
result = t.get_nns_by_item(0, 5)  # Get 5 nearest neighbors for item 0
print(result)

#ScaNN (Scalable Nearest Neighbors)
 
import scann
# Example setup for ScaNN
dataset = np.random.random((1000, 128))  # Random dataset
scann_index = scann.scann_ops_pybind.builder(dataset, 10, "dot_product").build()

# Querying
query = np.random.random(128)
neighbors, distances = scann_index.search(query, 5)
print(neighbors)
"""
ANN Search Benefits:
Speed: ANN techniques are significantly faster than exact search methods.
Scalability: They can handle large datasets with millions of vectors.
Efficiency: These methods reduce memory usage and computational costs.
Challenges with ANN:
Approximation: The results may not always be 100% accurate.
Choice of Algorithm: Different ANN algorithms (e.g., KD-trees, LSH) may perform better with different types of data or queries.
2. Inverted Indexing
Inverted indexing is a classic technique used in information retrieval, particularly for text data. 
The idea is to create a "dictionary" that maps each unique term to the documents (or items) that contain it. 
This allows for fast keyword-based search, as the search engine can quickly look up the term and retrieve a list of relevant documents.

How to Works 
Tokenization: First, the text data is tokenized into words or terms.
Indexing: For each term, the inverted index stores the list of document IDs (or positions) where the term appears.
Querying: When a query is made, the system looks up the terms in the inverted index and retrieves the relevant documents.
Example:
Consider a small dataset of product descriptions:

Doc 1: "Apple iPhone 12"
Doc 2: "Samsung Galaxy S21"
Doc 3: "Apple iPhone 13"
The inverted index would look like this:

rust
Copy code
'Apple' -> [1, 3]
'iPhone' -> [1, 3]
'12' -> [1]
'Samsung' -> [2]
'Galaxy' -> [2]
'S21' -> [2]
'13' -> [3]
To search for products related to Apple, the query would return documents 1 and 3.

Combining Inverted Indexing with Other Techniques (Hybrid Search):
Hybrid Search: In modern applications, inverted indexing can be combined with vector-based similarity search.
 For example, a search engine may first use inverted indexing to quickly narrow down the relevant documents based on keywords,
 then use a vector search to rank those results based on semantic similarity.
Example of Hybrid Search:
Imagine you have both the inverted index and an ANN index for product descriptions.
 A user searches for “Apple iPhone”. The system first uses the inverted index to find all documents containing "Apple" and "iPhone". 
 Then, it applies a vector-based similarity search on the subset of documents to rank them by relevance  

Benefits of Inverted Indexing:
Efficiency: Very fast lookups for keyword-based queries.
Scalability: Handles very large document collections.
Simplicity: Easy to implement and understand.
Challenges with Inverted Indexing:
Complexity for Advanced Queries: Inverted indices are effective for exact matches but may struggle with more complex queries like fuzzy matches or semantic searches.
Storage: Inverted indices can grow large, especially for large datasets with a rich vocabulary.
"""