import sys
import os
import torch

# Added source and model directories to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))

# Imported required functions and model classes
from utils import load_data_1m, clean_data, merge_datasets, preprocess_genres, get_id_mappings, get_movie_details
from content_based_model import ContentBasedRecommender
from collaborative_filtering_model import CollaborativeFilteringRecommender
from matrix_factorization_model import MatrixFactorizationRecommender
from neural_model import NeuralRecommender, recommend_movies

def combine_recommendations(list_of_lists):
    # Combined multiple recommendation lists by assigning higher weights to top-ranked movies
    from collections import Counter
    c = Counter()
    for recs in list_of_lists:
        for i, movie in enumerate(recs):
            c[movie] += (len(recs) - i)
    # Returned top 10 most frequently and highly ranked movies
    return [movie for movie, _ in c.most_common(10)]

def main():
    # Took user ID input from the user
    user_id = int(input("Enter user ID: ").strip())

    # Loaded and preprocessed MovieLens 1M dataset
    movies, ratings = load_data_1m()
    clean_movies = clean_data(movies)
    merged = merge_datasets(clean_movies, ratings)
    genre_features, _ = preprocess_genres(clean_movies)
    user2idx, idx2user = get_id_mappings(merged, 'userId')
    movie2idx, idx2movie = get_id_mappings(merged, 'movieId')

    print(f"\nRecommendations for User ID {user_id}:\n")

    # Generated content-based recommendations
    cb_model = ContentBasedRecommender(clean_movies)
    cb_recs = cb_model.recommend(user_id)

    # Generated collaborative filtering recommendations
    cf_model = CollaborativeFilteringRecommender(ratings)
    cf_rec_ids = cf_model.recommend(user_id)
    cf_recs = get_movie_details(cf_rec_ids, clean_movies)

    # Loaded pre-trained Matrix Factorization model and generated recommendations
    mf_model = MatrixFactorizationRecommender(len(user2idx), len(movie2idx))
    mf_model.load_state_dict(torch.load('mf_model.pt'))  # Assumed model weights were pre-trained and saved
    mf_model.eval()
    mf_rec_ids = mf_model.recommend(user_id, user2idx, movie2idx, idx2movie)
    mf_recs = get_movie_details(mf_rec_ids, clean_movies)

    # Loaded pre-trained Neural Recommender model and generated neural collaborative filtering recommendations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    neural_model = NeuralRecommender(len(user2idx), len(movie2idx), genre_features.shape[1]).to(device)
    neural_model.load_state_dict(torch.load('neural_model_best.pt', map_location=device))
    neural_model.eval()
    neural_rec_ids = recommend_movies(neural_model, user_id, user2idx, movie2idx, idx2movie, genre_features, device=device)
    neural_recs = get_movie_details(neural_rec_ids, clean_movies)

    # Printed results from all individual recommendation methods
    print("Content-Based:\n", cb_recs)
    print("Collaborative Filtering:\n", [title for title, _ in cf_recs])
    print("Matrix Factorization:\n", [title for title, _ in mf_recs])
    print("Neural Collaborative Filtering:\n", [title for title, _ in neural_recs])

    # Combined all recommendation outputs to produce hybrid results
    combined = combine_recommendations([
        cb_recs,
        [title for title, _ in cf_recs],
        [title for title, _ in mf_recs],
        [title for title, _ in neural_recs]
    ])
    print("\nCombined Hybrid Recommendations:\n", combined)

if __name__ == "__main__":
    main()
