import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def load_data_1m():
    # Loaded MovieLens 1M ratings data from file
    ratings = pd.read_csv(
        'data/ml-1m/ratings.dat',
        sep='::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    # Loaded MovieLens 1M movies data from file
    movies = pd.read_csv(
        'data/ml-1m/movies.dat',
        sep='::',
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='latin-1'
    )
    return movies, ratings

def clean_data(movies):
    # Replaced missing genres with empty lists and split genres by '|'
    movies.loc[:, 'genres'] = movies['genres'].fillna('').apply(lambda x: x.split('|') if isinstance(x, str) else [])
    return movies

def merge_datasets(movies, ratings):
    # Merged movies and ratings datasets on 'movieId' column
    merged = ratings.merge(movies, on='movieId', how='inner')
    return merged

def preprocess_genres(movies):
    # Converted list of genres into binary features using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres'])
    # Created DataFrame with genre features for each movie
    genre_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=movies.index)
    return genre_df, mlb

def get_id_mappings(df, col):
    # Extracted unique IDs from the given column
    unique = df[col].unique()
    # Created mappings between original IDs and numeric indices
    id2idx = {id_: idx for idx, id_ in enumerate(unique)}
    idx2id = {idx: id_ for idx, id_ in enumerate(unique)}
    return id2idx, idx2id

def create_user_name_mappings(ratings):
    # Created fake usernames for user IDs for better readability
    unique_users = ratings['userId'].unique()
    user_id_to_name = {uid: f"user_{uid}" for uid in unique_users}
    name_to_user_id = {v: k for k, v in user_id_to_name.items()}
    return user_id_to_name, name_to_user_id

def get_movie_details(movie_ids, movies_df):
    # Filtered movies DataFrame for given movie IDs
    filtered = movies_df[movies_df['movieId'].isin(movie_ids)].copy()
    # Ensured movieId column was integer and matched input order
    filtered['movieId'] = filtered['movieId'].astype(int)
    filtered = filtered.set_index('movieId').loc[movie_ids].reset_index()
    # Combined genres list into a single string for readability
    genres_list = filtered['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else str(x))
    # Returned list of tuples (title, genres)
    details = list(zip(filtered['title'], genres_list))
    return details

def rmse(y_true, y_pred):
    # Calculated Root Mean Squared Error between true and predicted ratings
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
