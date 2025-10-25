import sys
import os

# Added parent directories to Python path so the project modules could be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import streamlit as st
import torch

# Imported different recommendation models
from content_based_model import ContentBasedRecommender
from collaborative_filtering_model import CollaborativeFilteringRecommender
from matrix_factorization_model import MatrixFactorizationRecommender
from neural_model import NeuralRecommender, recommend_movies

# Imported utility functions for data loading and processing
from utils import load_data_1m, clean_data, merge_datasets, preprocess_genres, get_id_mappings, get_movie_details


def set_blur_background():
    # This function was used to set a blurred background and custom theme for the Streamlit app
    st.markdown(
        """
        <style>
        body, [data-testid="stAppViewContainer"] {
            background: none !important;
        }
        #blur-bg {
            position: fixed;
            top:0; left:0; width:100vw; height:100vh; z-index:-1;
            background-image: url('https://cdn.theatlantic.com/thumbor/VtMJhRgQzcQRfjcd_R2nGOtiXgA=/0x0:2601x1463/960x540/media/img/mt/2020/03/GettyImages_120685930/original.jpg');
            background-size: cover;
            background-position: center center;
            filter: blur(px) brightness(0.72);
        }
        .stApp {
            background: none !important;
        }
        h1, h2, h3, h4, h5, h6, .stMarkdown, p, div, .stTextInput, .stNumberInput input, label, .stButton > button {
            color: #fff !important;
            font-weight: bold !important;
            text-shadow: 0 1px 18px #222;
        }
        .stTextInput > div > input, .stNumberInput input {
            background: #222a;
            color: #fff !important;
            font-weight: bold !important;
        }
        .stButton > button {
            background-color: #fff;
            color: #111;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton > button:hover {
            background-color: #fc0;
            color: #111;
        }
        </style>
        <div id="blur-bg"></div>
        """,
        unsafe_allow_html=True
    )


def combine_recommendations(list_of_lists):
    # This function combined movie recommendations from multiple models by assigning weighted scores
    from collections import Counter
    c = Counter()
    for recs in list_of_lists:
        for i, movie in enumerate(recs):
            c[movie] += (len(recs)-i)
    # Returned top 10 most common recommended movies
    return [movie for movie, _ in c.most_common(10)]


@st.cache_data
def load_all_data():
    # Loaded and cleaned MovieLens 1M dataset
    movies, ratings = load_data_1m()
    clean_movies = clean_data(movies)
    merged = merge_datasets(clean_movies, ratings)
    
    # Processed genres into numerical features
    genre_features, _ = preprocess_genres(clean_movies)
    
    # Created user and movie ID mappings for model training and predictions
    user2idx, idx2user = get_id_mappings(merged, 'userId')
    movie2idx, idx2movie = get_id_mappings(merged, 'movieId')
    
    # Returned all prepared data
    return movies, clean_movies, ratings, merged, genre_features, user2idx, idx2user, movie2idx, idx2movie


def main():
    # Set custom background
    set_blur_background()

    # Displayed app title and subtitle with HTML styling
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:18px;'>
            <span style="font-size:2.3rem; font-weight: bold;">🎭 MovieGenius: The Hybrid Engine</span>
            <div style='font-size:1.2rem; font-weight: bold; margin-top:11px;'>
                Smart movie picks, blending classic and AI-powered recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # Loaded all data for recommendation models
    (movies, clean_movies, ratings, merged, genre_features, user2idx, idx2user, movie2idx, idx2movie) = load_all_data()

    # Took user input for User ID
    user_id = st.number_input("Enter your User ID:", min_value=1, step=1, value=1)

    # Generated recommendations when user clicked the button
    if st.button('Get Recommendations'):
        # Initialized all models
        cb_model = ContentBasedRecommender(clean_movies)
        cf_model = CollaborativeFilteringRecommender(ratings)
        mf_model = MatrixFactorizationRecommender(len(user2idx), len(movie2idx))

        # Tried to load pre-trained matrix factorization model
        try:
            mf_model.load_state_dict(torch.load('mf_model.pt'))
            mf_model.eval()
        except Exception as e:
            st.warning(f"Matrix Factorization model weights not found. Please train it. {e}")

        # Checked if CUDA (GPU) was available and loaded neural model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        neural_model = NeuralRecommender(len(user2idx), len(movie2idx), genre_features.shape[1]).to(device)

        # Tried to load pre-trained neural model
        try:
            neural_model.load_state_dict(torch.load('neural_model_best.pt', map_location=device))
            neural_model.eval()
        except Exception as e:
            st.warning(f"Neural model weights not found. Please train it. {e}")

        # Generated recommendations from each model
        cb_recs = cb_model.recommend(user_id)
        cf_rec_ids = cf_model.recommend(user_id)
        cf_recs = get_movie_details(cf_rec_ids, clean_movies)
        mf_rec_ids = mf_model.recommend(user_id, user2idx, movie2idx, idx2movie)
        mf_recs = get_movie_details(mf_rec_ids, clean_movies)
        neural_rec_ids = recommend_movies(neural_model, user_id, user2idx, movie2idx, idx2movie, genre_features, device=device)
        neural_recs = get_movie_details(neural_rec_ids, clean_movies)

        # Combined recommendations from all models
        combined = combine_recommendations([
            cb_recs, 
            [t for t, _ in cf_recs], 
            [t for t, _ in mf_recs], 
            [t for t, _ in neural_recs]
        ])

        # Displayed final top recommended movies
        st.markdown("<h3 style='text-align:center; margin-top:32px;'>🌟 Your MovieGenius Picks</h3>", unsafe_allow_html=True)
        for title in combined:
            st.markdown(f"<div style='margin-bottom:8px; font-size:1.19rem;'>🎬 {title}</div>", unsafe_allow_html=True)


# Ran the Streamlit app
if __name__ == "__main__":
    main()
