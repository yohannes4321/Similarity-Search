"""Jaccard Similarity  
The Jaccard Similarity measures the similarity between two sets by comparing the number of elements they 
share to the total number of unique elements in both sets. Mathematically, it is given by the formula:
Jaccard Similarity = |Intersection| / |Union|"""

def jaccard_similarity(set1, set2):
    # Calculate intersection and union
    intersection = len(set1.intersection(set2))  # common elements
    union = len(set1.union(set2))  # all unique elements
    
    # Calculate Jaccard Similarity
    if union == 0:  # avoid division by zero if both sets are empty
        return 0
    return intersection / union

# Example usage
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print("Jaccard Similarity:", jaccard_similarity(set1, set2))
"""
set1 and set2: These are the two sets you want to compare.
set1.intersection(set2): Finds the elements that are common to both sets.
set1.union(set2): Combines all unique elements from both sets.
Jaccard Similarity Calculation: The similarity is the ratio of the size of the intersection to the size of the union.
Example Walkthrough:
 """
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
"""
The intersection of set1 and set2 is {3, 4} with a size of 2.
The union of set1 and set2 is {1, 2, 3, 4, 5, 6} with a size of 6.
So, the Jaccard Similarity is:

 
 
Jaccard Similarity: 0.3333333333333333
"""





