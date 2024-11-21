Nearest Neighbor search 
proximity search is the optimization problem of finding 
the point in a given set that is closest (most similar to given point )


measure similarity and dissimilarity between two vectors

Distance and similarity metric

Euclidean distance
measure stright line distance 

cosine similarity measure the cosine angle between two vectors 


steps
 1 for a given query vector calculate the distance and similarity to every other vectors 
 sort the vectors based on distance or similarity

2 sort the result and return the closest vectors(nearest neighbor)

 the is brute force techinques so there is method to optimize this techinques 

 1 KD Tress (low dimenstional data)
 2 Ball tress
 3 Approximate nearset Neighbour(ANN) method like FAISS

 Advantages 

 simplicity : easy to implemint and understand
 versatility : works with any type of data where distance metric is defind 
 high accuracy : find exact and approximate neighbors based on defined metric 
 