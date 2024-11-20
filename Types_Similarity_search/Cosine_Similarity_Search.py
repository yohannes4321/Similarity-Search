"""Cosine Similarity Search

measure of similarity  between two non zero vectors defined in inner product space 
dot product of vectors divided by the product of the vectors with absolute value 
    
  
For example
1 two proportional vectors ( two identical vectors )have cosine similiarity of 1
2 two opposite vectors have cosine similarity of 0
3 two orthogonal vectors have cosine similarity of 0

cosine similarity = A.B/ |A|*|B|
which is summition of Ai.Bi / squareroot(sum`Ai^2) * sqaureroot(sum`Bi^2))   
the range is from -1 to 1


Advantages of Cosine similarity Search 


1 Ignore the magnitude  and focus on the direction of the vectors 
which means it does not matter how large and small the vectors is 
what matter most is the direction 

2 Efficient for High Dimensional Data
cosine similiraty is very efficient and works well even if the vectors are sparce(
    for example many zeros  in ttext representations
) 

3 result is easy to interpret:
the output ranges from -1 to 1 
1 means vectors point to the same direction 
0 vectors are orthogonal (no similarity)
-1 vetors point in opposite direction (completely dissimilar)


DisAdvantages of Cosine similarity Search

1 Does not measure distance
it is not distance metric two vectors have high cosine similarity even if large Euclidean distance 
in space 

2 Limited to Linear Relationship it doesnot capture complex pattern non linear similarity .

"""
import numpy as np
def cosine_similar(x,y):
    dot=np.dot(x,y)
    norm1=np.linalg.norm(x)
    norm2=np.linalg.norm(y)
    return dot/(norm1*norm2)

x = np.array([1, 2, 3])   
y= np.array([4, 5, 6])  

similarity = cosine_similar(x,y)   
print(f"cosine similarity: {similarity:.4f}")

