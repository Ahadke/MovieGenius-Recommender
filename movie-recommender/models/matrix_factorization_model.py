import torch
import torch.nn as nn

class MatrixFactorizationRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        # Created embedding layers for users and movies to learn latent features
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

    def forward(self, user_indices, movie_indices):
        # Fetched user and movie embeddings
        user_embeds = self.user_embedding(user_indices)
        movie_embeds = self.movie_embedding(movie_indices)
        
        # Computed dot product between user and movie embeddings to predict rating
        return (user_embeds * movie_embeds).sum(1)

    def recommend(self, user_id, user2idx, movie2idx, idx2movie, top_n=10):
        # Converted user ID to corresponding index
        user_idx = user2idx.get(user_id)
        if user_idx is None:
            return []

        # Collected all movie indices
        movie_indices = list(movie2idx.values())
        
        # Created tensors for user and movies
        user_tensor = torch.LongTensor([user_idx] * len(movie_indices))
        movie_tensor = torch.LongTensor(movie_indices)
        
        # Predicted scores for all movies for this user without computing gradients
        with torch.no_grad():
            scores = self(user_tensor, movie_tensor).numpy()
        
        # Picked top N highest scoring movies
        top_indices = scores.argsort()[-top_n:][::-1]
        
        # Mapped movie indices back to their original movie IDs
        recommended_movie_ids = [idx2movie[movie_indices[i]] for i in top_indices]
        return recommended_movie_ids
