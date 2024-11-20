"""Euclidean Distance Search 

Euclidean distance metric calcultes stright line shortes distance 
between two points in n-dimensional space

the distance between two vectors and x,y,z and square and add them finally sqrt
formula for Euclidean distance

d(p,q) = sqrt((x1 - y1)^2 + (x2 - y2)^2)

Advantages

1 its StrightForward and easy to interpret
2 Effective for low dimensional data
3 standard for many applications like k Nearst Neighbors
    

Disadvantages
1 sensitive to scale 
so we have to normilize or standardize the data
2 senstive to outliers
large outliers can skew the results significantly 
3 ignores Directions
it only considers magnitude not directionality 
    """
import faiss
import numpy as np
database=np.array([[1,2,3],[4,5,6],[7,8,9]])
index=faiss.IndexFlatL2(3)
query_vector=np.array([[3,3,3]],dtype=np.float32)
distance,indices=index.search(query_vector,3)
print(f"Nearest neighbor index: {indices[0][0]}")
print(f"Euclidean distance: {distance[0][0]}")
print(f" 2ND Nearest neighbor index: {indices[0][1]}")
print(f" 2ND Euclidean distance: {distance[0][1]}")

print(f"3RD Nearest neighbor index: {indices[0][2]}")
print(f"3RD Euclidean distance: {distance[0][2]}")