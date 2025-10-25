from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedRecommender:
    def __init__(self, movies):
        # Stored a copy of the movie dataset to work with
        self.movies = movies.copy()
        
        # Created a TF-IDF vectorizer to convert movie genres into numerical feature vectors
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Transformed genre lists into text and then into TF-IDF matrix form
        self.tfidf_matrix = self.tfidf.fit_transform(
            movies['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
        )
        
        # Calculated cosine similarity between all movies based on their genre vectors
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def recommend(self, user_id, top_n=10):
        # Used a simple fallback method — recommended top popular movies 
        # (In advanced versions, this would use the user's liked movies to find similar ones)
        recommended = self.movies['title'].head(top_n).tolist()
        
        # Returned the list of recommended movie titles
        return recommended
