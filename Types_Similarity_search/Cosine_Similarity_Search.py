import numpy as np
"""
Cosine similarity is a widely used metric for measuring the similarity between two vectors,
 particularly in text analysis, such as in natural language processing (NLP) tasks like text mining, sentiment analysis, and 
 document clustering. Here's an explanation of how to compute cosine similarity, along with Python code to demonstrate it.

cosine similarity(|X,Y|)=x.y /(|x||y|)"""
 

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    # Compute the dot product of vec1 and vec2
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitude (norm) of vec1 and vec2
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim

# Example vectors (these can be word embeddings or any feature vectors)
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

# Compute and print cosine similarity
similarity = cosine_similarity(vec1, vec2)
"""
Dot Product: np.dot(vec1, vec2) computes the dot product between the two vectors.
Magnitude (Norm): np.linalg.norm(vec) computes the Euclidean norm (magnitude) of a vector.
Cosine Similarity Calculation: The formula is applied by dividing the dot product by the product of the magnitudes of the two vectors.
Example Vectors: We provide two example vectors, vec1 and vec2. In real-world applications, these could be text representations such as word embeddings (e.g., from Word2Vec, GloVe, or BERT).

Result: The function returns the cosine similarity value, which ranges from -1 (completely opposite vectors), 0 (orthogonal vectors), to 1 (identical vectors).

Output
For the vectors vec1 = [1, 2, 3] and vec2 = [4, 5, 6], the output will be:

Similarity: The higher the cosine similarity (closer to 1), the more similar the vectors are. If the cosine similarity is 0,
the vectors are orthogonal, indicating no similarity.
Negative Similarity: A negative cosine similarity means the vectors are opposite, i.e., their angle is greater than 90°.
"""



