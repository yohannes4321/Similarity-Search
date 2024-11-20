"""The Manhattan Distance (also known as the Taxicab or City Block Distance) is a metric that calculates the distance between two points in a grid-based system, like the streets of Manhattan, where movement is restricted to horizontal and vertical paths. It is based on the sum of the absolute differences between corresponding coordinates of two points.

Formula for Manhattan Distance
The Manhattan distance between two points (x1, y1) and (x2, y2) is given by:

d(x1, y1, x2, y2) = |x1 - x2| + |y1 - y2|
In other words, it is the sum of the absolute differences between the x and y coordinates of the two points.

 
Here's how you can calculate the Manhattan distance between two vectors in Python:

 """
def manhattan_distance(vector1, vector2):
    # Check if vectors have the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same number of dimensions.")
    
    # Calculate the Manhattan distance by summing the absolute differences of corresponding elements
    distance = sum(abs(v1 - v2) for v1, v2 in zip(vector1, vector2))
    return distance

# Example usage
vector1 = [1, 2, 3]
vector2 = [4, 0, 3]

print("Manhattan Distance:", manhattan_distance(vector1, vector2))
"""
manhattan_distance function: This function takes two vectors (lists) as input.
Validation: It checks if the vectors are of the same length. If they are not, it raises a ValueError.
Distance Calculation: It calculates the Manhattan distance by iterating over the corresponding elements of the two vectors using zip(), taking the absolute difference, and summing them up.
Example Usage: For vectors [1, 2, 3] and [4, 0, 3], the Manhattan distance is calculated as 5.

Manhattan Distance: 5
"""