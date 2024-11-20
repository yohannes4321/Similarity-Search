"""Content-based filtering

uses item features to recommend items simmilar to what the User
likes,based from the previous actions of the User.

This feature can be converted into numercal representation 


 1 TF-IDF(Term Frequency Inverse Document Frequency)
how important a word is in a document
TERM FREQUENCY:
how frequent a term appear in frequancy which means higher frequancy means 
higher importance

inverse Doucment FREQUENCY:
How important a term in entire corpus words that appear in many documents 
are lower importance 
words that is found is fewer documents are more important 


Word Embeddings

capture semantic relationships between words, allowing models to 
understand and represent words in a continuous vector space where 
similar words are close to each other


Common algorthms for generating word embeddings:

1. Word2Vec
2. GloVe

Advantages:
1. Captures semantic relationships between words
2 Dense and more effient than tradional one hot Encoding 
DisAdvantages
1,pre trained embedding may not work for domain specfic words
2 doesnot capture word sense disambiugation"bank" could mean a financial 
institution or the side of a river).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'Genre': ['Action, Thriller', 'Action, Adventure', 'Drama, Romance', 'Action, Thriller'],
    'Director': ['Director X', 'Director Y', 'Director Z', 'Director X'],
    'Actors': ['Actor 1, Actor 2', 'Actor 3, Actor 4', 'Actor 5, Actor 6', 'Actor 1, Actor 7']
}

 
df = pd.DataFrame(data)

 
df['Content'] = df['Genre'] + ' ' + df['Director'] + ' ' + df['Actors']

 
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

 
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Content'])

 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

 
def recommend_movie(movie_title):
     
    idx = df.index[df['Movie'] == movie_title].tolist()[0]
 
    sim_scores = list(enumerate(cosine_sim[idx]))

 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

 
    sim_scores = sim_scores[1:4]

 
    movie_indices = [i[0] for i in sim_scores]

    
    return df['Movie'].iloc[movie_indices]

 
recommended_movies = recommend_movie('Movie A')
print("Recommended Movies:", recommended_movies)

