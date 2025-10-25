import sys
import os

# Added the parent directory to sys.path so imports worked
# This ensured that the script could correctly import modules from the 'src' folder and parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imported required libraries
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Imported custom utility functions and model class
from utils import load_data_1m, clean_data, merge_datasets, preprocess_genres, get_id_mappings
from matrix_factorization_model import MatrixFactorizationRecommender

def main():
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Loaded MovieLens 1M dataset (movies and ratings)
    movies, ratings = load_data_1m()

    # Step 2: Cleaned movie data (handled missing or invalid entries)
    clean_movies = clean_data(movies)

    # Step 3: Merged movies and ratings data into one dataframe
    merged = merge_datasets(clean_movies, ratings)

    # Step 4: Created mappings between user/movie IDs and numeric indices
    user2idx, idx2user = get_id_mappings(merged, 'userId')
    movie2idx, idx2movie = get_id_mappings(merged, 'movieId')

    # Step 5: Initialized Matrix Factorization model
    model = MatrixFactorizationRecommender(len(user2idx), len(movie2idx)).to(device)

    # Step 6: Defined optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()

    # Step 7: Added placeholder section for training logic
    # (To be replaced later with actual data batching and optimization loop)
    # TODO: Implemented realistic training loop with data batches

    # Step 8: Saved the trained model weights
    torch.save(model.state_dict(), 'mf_model.pt')
    print("Matrix factorization model saved.")

# Marked the entry point of the script
if __name__ == "__main__":
    main()
