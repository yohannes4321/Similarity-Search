"""
Hamming Distance is a metric that quantifies the difference between two strings of equal length. Specifically, it counts the number of positions at which the corresponding symbols (characters or bits) are different. It is widely used in error detection, coding theory, and in applications where binary sequences or strings need to be compared for similarity.

Hamming Distance Formula
Example:
For two binary strings:

 
a = 1011101
b = 1001001
Comparing them bit by bit:

1 != 1 → 0 (no difference)
0 != 0 → 0 (no difference)
1 != 0 → 1 (difference)
1 != 1 → 0 (no difference)
1 != 0 → 1 (difference)
0 != 0 → 0 (no difference)
1 != 1 → 0 (no difference)
Thus, the Hamming distance between the two strings is 2 (there are two positions where the strings differ).

How Hamming Distance Search Works
In the context of searching:

Hamming Distance Search is useful when you want to find strings that differ from a given query string by a specific number of bits or characters.
This can be used in error detection, DNA sequence comparison, image processing, or finding closely related strings in databases.
For example, a search query could be a binary sequence, and you may want to find all database entries that differ by at most 1 bit. This is helpful in error correction codes, where you need to detect and correct minor errors in the data.
 
"""
# Function to calculate Hamming distance between two strings
def hamming_distance(str1, str2):
    # Ensure the strings are of the same length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    
    # Calculate the number of positions where the characters are different
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Function to perform a Hamming Distance search on a list of strings
def hamming_search(query, dataset, max_distance):
    # List to store strings that are within the allowed Hamming distance
    results = []
    
    # Check each string in the dataset
    for data in dataset:
        # Calculate the Hamming distance between the query and each dataset string
        distance = hamming_distance(query, data)
        
        # If the distance is within the max allowed, add to results
        if distance <= max_distance:
            results.append(data)
    
    return results

# Example usage
dataset = ["1011101", "1001001", "1110001", "0111010", "1010111"]
query = "1011001"
max_distance = 2

# Perform the search
results = hamming_search(query, dataset, max_distance)

print(f"Strings within a Hamming distance of {max_distance} from '{query}':")
for result in results:
    print(result)
"""
The hamming_distance function compares two strings of equal length and counts the positions where they differ.
It raises an exception if the strings are of different lengths, as Hamming Distance is only defined for strings of the same length.
Hamming Search:

The hamming_search function accepts a query string and a dataset (list of strings).
It iterates over the dataset, calculates the Hamming distance for each string in the dataset relative to the query string, and collects strings whose Hamming distance is less than or equal to a specified max_distance.
Example:

The dataset contains a list of binary strings.
The query is another binary string.
The search returns all strings in the dataset that differ from the query by at most 2 positions
 


Strings within a Hamming distance of 2 from '1011001':
1011101
1001001
1010111
In this example, the query string 1011001 differs from the strings 1011101, 1001001, and 1010111 by 2 or fewer bits, so they are returned as part of the result.
 
"""

