import numpy as np

"""Euclidean distance
is the straight-line distance between two points 
 in a multi-dimensional space. The Euclidean Distance Search is a method used to find the 
 nearest points (or vectors) to a given point based on their Euclidean distances.
d(p,q)=√(x(p)-x(q))2+(y(p)-y(q))2+(z(p)-z(q))2
The Euclidean distance is computed between the query vector and each data point.
The distances indicate how "close" or "similar" each data point is to the query. A smaller distance implies a higher similarity.
Returning Nearest Neighbors:

Once distances are calculated, data points are ranked based on their distances from the query.
The nearest points (i.e., with the smallest distances) are returned as the search results.

 
Steps:
Vector Representation: Data points are represented as vectors in an n-dimensional space.
Query Point: A query point is also represented as a vector.
Distance Calculation: Euclidean distance is calculated between the query point and each data point.
Nearest Neighbors: Data points are ranked based on their Euclidean distances from the query point, and the nearest neighbors are returned.
Code Implementation:
"""
 
def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

# Sample data points (vectors) representing user profiles
data_points = [
    {"id": 1, "profile": [25, 50000, 200]},   # Age, Income, Spending Score
    {"id": 2, "profile": [30, 55000, 250]},
    {"id": 3, "profile": [22, 45000, 180]},
    {"id": 4, "profile": [35, 60000, 300]},
    {"id": 5, "profile": [28, 52000, 220]},
]

# Query point (e.g., a user's query profile)
query_point = [28, 52000, 220]

# Calculate the Euclidean distance between the query point and each data point
distances = []
for data in data_points:
    distance = euclidean_distance(query_point, data["profile"])
    distances.append({"id": data["id"], "distance": distance})

# Sort the distances to get the nearest neighbors
sorted_distances = sorted(distances, key=lambda x: x["distance"])

# Display the nearest neighbors
print("Nearest Neighbors (sorted by Euclidean Distance):")
for neighbor in sorted_distances:
    print(f"ID: {neighbor['id']}, Distance: {neighbor['distance']:.4f}")
"""
The function euclidean_distance(p, q) computes the straight-line distance between two vectors p and q.
This is done by calculating the squared differences between corresponding elements in the vectors, summing them up, and then taking the square root of the result.
Sample Data Points:

data_points represents a set of user profiles, each with three features: age, income, and spending score.
Each profile is a vector, and these vectors are used to calculate the Euclidean distance to the query vector.
Query Point:

query_point represents the vector of the query (e.g., the features of a user or item you're searching for).
Calculating Distances:

For each data point in the dataset, the Euclidean distance to the query point is calculated using the euclidean_distance function.
The results are stored in the distances list, which also includes the ID of each data point for identification.
Sorting:

The distances are sorted in ascending order using the sorted function. This ensures that the nearest neighbors (smallest distance) come first.
Displaying Results:
Nearest Neighbors (sorted by Euclidean Distance):
ID: 5, Distance: 0.0000
ID: 2, Distance: 22.6274
ID: 1, Distance: 35.3553
ID: 3, Distance: 49.4977
ID: 4, Distance: 70.7107
Key Points:
Euclidean Distance is used to measure how far apart two vectors are in a multi-dimensional space. In this case,
 the space represents user features like age, income, and spending score.
The smaller the distance, the closer the data point is to the query point.
This method is very effective in nearest neighbor searches, clustering, and many machine learning tasks
 where you need to find data points that are similar to a given query."""