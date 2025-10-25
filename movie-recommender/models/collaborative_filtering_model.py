import pandas as pd
from sklearn.neighbors import NearestNeighbors

class CollaborativeFilteringRecommender:
    def __init__(self, ratings):
        # Created a user-item matrix where rows represent users, columns represent movies, and values are ratings
        self.user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Initialized a K-Nearest Neighbors model using cosine similarity to find similar users
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        
        # Trained (fit) the KNN model on the user-item matrix
        self.knn.fit(self.user_item_matrix)

    def recommend(self, user_id, top_n=10):
        # Checked if the given user ID exists in the dataset
        if user_id not in self.user_item_matrix.index:
            return []

        # Found the 5 most similar users (excluding the user themself)
        distances, indices = self.knn.kneighbors(
            self.user_item_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=6
        )

        # Extracted similar user IDs
        similar_users = self.user_item_matrix.index[indices.flatten()[1:]]

        # Collected the movies already rated by the target user
        user_movies = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)

        # Created a set to store recommended movies
        recs = set()

        # Gathered movies that similar users rated highly (rating > 4) but the target user has not watched
        for u in similar_users:
            recs.update(
                self.user_item_matrix.loc[u][self.user_item_matrix.loc[u] > 4].index.difference(user_movies)
            )
            # Stopped once enough (top_n) recommendations were collected
            if len(recs) >= top_n:
                break

        # Returned the top N recommended movie IDs
        return list(recs)[:top_n]
