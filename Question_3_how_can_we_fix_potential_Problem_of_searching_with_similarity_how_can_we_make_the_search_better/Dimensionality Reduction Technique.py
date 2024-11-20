import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
"""
PCA-indexing (Piecewise Constant Approximation)

is a technique proposed for efficient similarity search in large time series databases.
It leverages dimensionality reduction to simplify the search process while retaining key features of the data. 
Let's break this down step-by-step, with a code example and a detailed explanation of how it works in practice.

Key Ideas Behind PCA-Indexing
Dimensionality Reduction:
Time series data can be very high-dimensional, which can make similarity searches slow and computationally expensive.
 PCA-indexing reduces the complexity by breaking the time series into smaller segments and representing each segment with a simple statistic 
 like the mean value of that segment.

Piecewise Constant Approximation (PCA):
The data is divided into fixed-length subsections (or "pieces"). Each piece is approximated by a constant value (usually the mean of the segment). 
This allows for a much simpler representation of the original data, making the similarity search faster since you're dealing with fewer dimensions.

Flexible Distance Measures:
One of the key advantages of PCA-indexing is its ability to support flexible distance metrics like weighted Euclidean distance,
 which is useful when some parts of the time series are more important than others.

Handling Variable Query Lengths:
PCA-indexing can efficiently handle queries of different lengths. It does this by padding shorter queries with zeros or 
truncating longer queries to match the indexed segments.

Speed and Scalability:
Compared to traditional methods like Discrete Fourier Transform (DFT) or wavelet transforms, PCA-indexing is much faster and simpler. 
It has been shown to outperform existing methods by orders of magnitude (up to 81 times faster in some cases).

Code Example: PCA-Indexing for Time Series Data
To illustrate how PCA-indexing works in practice, let’s create a simple Python code example where
 we apply PCA to a time series dataset, reduce its dimensionality, and perform a similarity search.

Step 1: Simulating Time Series Data
For the purpose of this example, we'll simulate some time series 
data representing different signals (e.g., sensor readings, stock prices).
"""
 
import numpy as np
import matplotlib.pyplot as plt

 

 
np.random.seed(0)
time = np.linspace(0, 10, 100)
signal1 = np.sin(time) + np.random.normal(0, 0.1, 100)
signal2 = np.cos(time) + np.random.normal(0, 0.1, 100)
signal3 = np.sin(time + 1) + np.random.normal(0, 0.1, 100)

# Plot the time series data
plt.plot(time, signal1, label="Signal 1 (Sine Wave)")




plt.plot(time, signal2, label="Signal 2 (Cosine Wave)")
plt.plot(time, signal3, label="Signal 3 (Sine Wave with Phase Shift)")
plt.legend()
plt.show()
"""
Step 2: Piecewise Constant Approximation (PCA)
Now we apply PCA (Piecewise Constant Approximation). We divide the time series into equal segments and compute the mean of each segment. This will be the reduced representation of the time series.

 """
def pca_indexing(signal, segment_length=10):
    
    n = len(signal)
    pca_rep = []
    
    # Segment the time series and compute the mean for each segment
    for i in range(0, n, segment_length):
        segment = signal[i:i+segment_length]
        segment_mean = np.mean(segment)
        pca_rep.append(segment_mean)
    
    return np.array(pca_rep)

# Apply PCA-indexing on the signals
pca_signal1 = pca_indexing(signal1, segment_length=10)
pca_signal2 = pca_indexing(signal2, segment_length=10)
pca_signal3 = pca_indexing(signal3, segment_length=10)

# Plot the PCA-reduced signals
plt.plot(pca_signal1, label="PCA Signal 1")
plt.plot(pca_signal2, label="PCA Signal 2")
plt.plot(pca_signal3, label="PCA Signal 3")
plt.legend()
plt.show()
"""
Step 3: Performing Similarity Search Using Euclidean Distance
Now that we have reduced the time series data, we can perform a similarity search. To do this, we’ll compute the Euclidean distance between the reduced representations of the signals.
""" 
 
# Combine the PCA signals into a matrix for easier computation
pca_signals = np.array([pca_signal1, pca_signal2, pca_signal3])

# Query: A new signal (query signal) we want to find similarities to
query_signal = np.sin(time + 0.5) + np.random.normal(0, 0.1, 100)
pca_query = pca_indexing(query_signal, segment_length=10)

# Compute the Euclidean distance between the query signal and the stored signals
distances = cdist([pca_query], pca_signals, metric='euclidean')

# Output the distances
print("Distances from query signal to other signals:", distances)
"""
Step 4: Interpreting the Results
The output will show the Euclidean distance between the query signal and each of the indexed signals. The signal with the smallest distance is the most similar to the query.

Advantages of PCA-Indexing
Speed: By reducing the dimensionality of the data, PCA-indexing allows for faster comparisons, making it well-suited for large-scale time series data.

Simplicity: PCA-indexing is straightforward to implement compared to more complex methods like wavelets or Fourier transforms, which require deeper mathematical understanding and more processing power.

Flexibility: It can support different distance measures, allowing customization of the search process. For example, you can give more weight to certain segments of the time series that may be more important for your application.

Scalability: PCA-indexing can handle large datasets efficiently, even when the dimensionality of the data is high.
"""
 