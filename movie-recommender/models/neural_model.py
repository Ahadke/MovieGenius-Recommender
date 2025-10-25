import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Neural Network model that learned user, movie, and genre features
class NeuralRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=50):
        super().__init__()
        # Created embeddings for users and movies to capture hidden preferences
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # Defined two fully connected layers for combining features
        self.fc1 = nn.Linear(embedding_dim * 2 + num_genres, 128)
        self.fc2 = nn.Linear(128, 1)

        # Added dropout and ReLU for better learning and regularization
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, user, movie, genre_vec):
        # Fetched user and movie embeddings
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)

        # Combined user, movie, and genre vectors into one input
        x = torch.cat([user_emb, movie_emb, genre_vec], dim=1)

        # Passed data through dense layers with activation and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # Predicted movie rating
        rating = self.fc2(x)
        return rating.squeeze()


# Custom dataset that prepared user, movie, rating, and genre data
class RatingsDataset(Dataset):
    def __init__(self, ratings_df, user2idx, movie2idx, genre_features):
        # Mapped user and movie IDs to index form
        self.user_ids = ratings_df['userId'].map(user2idx).values
        self.movie_ids = ratings_df['movieId'].map(movie2idx).values
        self.ratings = ratings_df['rating'].values.astype(float)
        self.genre_features = genre_features

    def __len__(self):
        # Returned total number of ratings
        return len(self.ratings)

    def __getitem__(self, idx):
        # Retrieved one training sample (user, movie, rating, genres)
        user = torch.tensor(self.user_ids[idx], dtype=torch.long)
        movie = torch.tensor(self.movie_ids[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float)
        genre_vec = torch.tensor(self.genre_features.iloc[self.movie_ids[idx]].values, dtype=torch.float)
        return user, movie, rating, genre_vec


# Trained the model using MSE loss and backpropagation
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for user, movie, rating, genre_vec in dataloader:
        user, movie, rating, genre_vec = user.to(device), movie.to(device), rating.to(device), genre_vec.to(device)
        optimizer.zero_grad()
        pred = model(user, movie, genre_vec)
        loss = F.mse_loss(pred, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * rating.size(0)
    return total_loss / len(dataloader.dataset)


# Evaluated model performance using RMSE metric
def evaluate_rmse(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for user, movie, rating, genre_vec in dataloader:
            user, movie, rating, genre_vec = user.to(device), movie.to(device), rating.to(device), genre_vec.to(device)
            pred = model(user, movie, genre_vec)
            preds.extend(pred.cpu().numpy())
            targets.extend(rating.cpu().numpy())
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()


# Generated top movie recommendations for a specific user
def recommend_movies(model, user_id, user2idx, movie2idx, idx2movie, genre_features, top_n=10, device='cpu'):
    model.eval()
    user_idx = user2idx.get(user_id)
    if user_idx is None:
        return []

    # Created tensors for all movies for the given user
    movie_indices = list(movie2idx.values())
    user_tensor = torch.tensor([user_idx] * len(movie_indices), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(movie_indices, dtype=torch.long).to(device)
    genre_vec_tensor = torch.tensor(genre_features.loc[movie_indices].values, dtype=torch.float).to(device)

    # Predicted scores and selected top N movies
    with torch.no_grad():
        preds = model(user_tensor, movie_tensor, genre_vec_tensor).cpu().numpy()

    top_indices = preds.argsort()[::-1][:top_n]
    recommendations = [idx2movie[movie_indices[i]] for i in top_indices]
    return recommendations
