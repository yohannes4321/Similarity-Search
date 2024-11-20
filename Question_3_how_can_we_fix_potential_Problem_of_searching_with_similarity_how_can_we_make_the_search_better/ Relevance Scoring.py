import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
"""
When building search systems, one of the most common challenges is ranking search results in a way that reflects not only 
the similarity between a query and available data but also the relevance or importance of those results. Relevance scoring allows you 
to weigh search results based on contextual features or user preferences. Let’s explore this concept in detail, and I’ll show 
you how you can implement relevance scoring using similarity-based scores combined with ranking models.

Key Concepts
Similarity-based scores: These scores typically come from calculating how similar an item is to a query, often using metrics
 like Euclidean distance, cosine similarity, or other distance functions. This measure of similarity helps rank the items in terms of closeness to the query.

 
User interaction history (e.g., clicks, likes, or purchases)
Context-specific weights (e.g., product category importance or time-sensitive queries)
User feedback (e.g., ratings or reviews)
Ranking models: Ranking models learn from data (often via machine learning) to predict the relevance of search results for a particular query.  
"""
 
# Sample product catalog (each product has a feature vector)
products = [
    {"id": 1, "name": "Smartphone", "features": [0.9, 0.3, 0.4, 0.8]},
    {"id": 2, "name": "Laptop", "features": [0.7, 0.8, 0.3, 0.6]},
    {"id": 3, "name": "Tablet", "features": [0.5, 0.4, 0.9, 0.7]},
    {"id": 4, "name": "Smartwatch", "features": [0.3, 0.6, 0.2, 0.5]},
]

# User query (features the user is interested in)
user_query = [0.8, 0.6, 0.5, 0.7]

# Convert products to DataFrame for easier manipulation
df = pd.DataFrame(products)
"""
Step 2: Compute Similarity
Now, we calculate the similarity between the user query and each product using cosine similarity.
 This will give us a similarity score for each product.
"""
 
# Compute cosine similarity between the user query and each product
product_features = np.array([product['features'] for product in products])
similarity_scores = cosine_similarity([user_query], product_features)

# Add similarity scores to the DataFrame
df['similarity_score'] = similarity_scores[0]
print(df[['id', 'name', 'similarity_score']])
"""
Step 3: Adding Relevance Weights
Let’s introduce relevance weights to adjust the ranking. We’ll simulate user feedback by assigning a relevance weight to each product. This weight could represent how much a user likes a product (e.g., based on past interactions or ratings).

 """
# Simulated user feedback (higher values indicate more relevance)
relevance_weights = {
    1: 0.9,   
    2: 0.8,  # Laptop is moderately relevant
    3: 0.5,  # Tablet is less relevant
    4: 0.6    
}

# Add relevance weights to the DataFrame
df['relevance_weight'] = df['id'].map(relevance_weights)
print(df[['id', 'name', 'similarity_score', 'relevance_weight']])
"""
Step 4: Combining Similarity with Relevance
To combine the similarity score and relevance weight, we can multiply the similarity by the relevance weight. This will adjust the ranking, giving more importance to items with higher relevance.

 """
# Combine similarity and relevance for final ranking score
df['final_score'] = df['similarity_score'] * df['relevance_weight']
print(df[['id', 'name', 'similarity_score', 'relevance_weight', 'final_score']])
"""
Step 5: Sorting the Results
Finally, we sort the products by the final score to get the ranked list of products based on both similarity and relevance.
"""
 
# Sort products by the final score (highest to lowest)
df_sorted = df.sort_values(by='final_score', ascending=False)
print(df_sorted[['id', 'name', 'final_score']])
"""
We calculate how similar each product is to the user’s query using cosine similarity. The closer the similarity score is to 1,
 the more similar the product is to the user query.

Relevance Weights:
We manually assign relevance weights based on hypothetical user feedback. These weights can be derived from various factors such as user ratings, 
previous purchases, or preferences. For instance, a product the user interacted with more frequently would have a higher weight.

Final Ranking:
We combine the similarity score with the relevance weight to adjust the ranking of the products. The final score represents how well a
 product matches the user's query, considering both similarity and relevance.

Sorting:
We sort the products based on their final score to present the most relevant and similar items at the top.
"""