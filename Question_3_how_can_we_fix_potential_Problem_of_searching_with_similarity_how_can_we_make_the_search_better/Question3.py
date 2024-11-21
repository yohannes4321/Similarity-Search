""" Semantic Search 

    Tradional search engines often rely on exact keyword searching but semantic 
    search works understanding the meaning of words and phrases 

    finding document with similar meaning 
        
        Word Embeddings
        Word2vec,Glove and BERT 
        THIS embedding represent words as vectors in high dimensional spaces 
        words with simialr meaning placed closed together 
        Advantage 
            Flexibility retrive documents even when query uses difrent pharsing 
        
            
            
            Accuracy : Reduce the problem of synonymes 
2 Vectorization 

    Text data is typically in raw data text so it must be changed numerical representation 
    vecterzation changes text to embeddings 

    BERT Glove FastTEXT


    bert forms contexual embedding it takes account of embedding context each word 


Preprocessing

    remove noise and irrelevant deitals to enhance the quality of search result 

    techniques 
        Stemming :Reduce words to there base root (runnig becomes run)
    Lemmatization : converts to there base form )('better to good ')
    Stop word removal : words like "the", "is", "and", removed 
Dimensionality Reducation 

reduce number of features in these vecotors while preserving much information 


    PCA(Principla componet analysis ) transforming data into smaller set of uncorrelated variables
    while retaing the variance 
    t-SNE A techinques desineed for visualizing high dimeniosnal data to lower dimension data 

Approximate Nearest Neighbor (ANN)
Faster alternative by sacrficaing a small amoutn of accuracy for speed 
   HNSW  ( Hierchical Naigable small world)  graph based approch for nearest neighbour 
search that is efficent and scalable 
Faiss



Fine Tuning and RANKING 
 RANK based of relevanve of feedback and ranking algorhims 
using user behaviour and query context and other singnal 

"""
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load a pretrained sentence transformer model (BERT-based)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example documents or search queries
documents = [
    "What is the capital of France?",
    "How to bake a cake?",
    "What is AI?",
    "How does machine learning work?",
    "France is in Europe."
]

# Generate embeddings for documents
embeddings = model.encode(documents)

# Convert to numpy array
embeddings = np.array(embeddings).astype('float32')


# Create an index using FAISS (it supports multiple index types, here using the flat index)
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)

# Add document embeddings to the FAISS index
index.add(embeddings)

# Example query to search
query = "Tell me about France's capital"
query_embedding = model.encode([query]).astype('float32')

# Search in FAISS (top 3 most similar documents)
D, I = index.search(query_embedding, k=3)  # D is distances, I is indices of results

# Output the results
print("Search Results:")
for i in range(len(I[0])):
    print(f"Document: {documents[I[0][i]]} - Distance: {D[0][i]}")
