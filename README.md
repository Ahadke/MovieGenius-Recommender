MovieGenius: Hybrid Movie Recommender System
This project demonstrates an advanced hybrid movie recommendation system built using the MovieLens 1M dataset. The dataset contains 1,000,209 anonymous ratings on approximately 3,900 movies by 6,040 users, providing a rich benchmark for recommender system development.

Project Overview
MovieGenius implements four distinct recommendation approaches and combines their strengths to offer personalized movie suggestions:

Content-Based Filtering: Recommends movies similar in genre and features to a user's preferences.

Collaborative Filtering: Finds similar users by rating patterns and suggests their favorites.

Matrix Factorization: Learns latent features jointly for users and items to predict preferences.

Neural Network Model: Uses deep learning to capture complex user-movie interactions and features.

The system is developed as an interactive web app using Streamlit, allowing users to enter their User ID and receive hybrid recommendations with a rich UI experience. The app features:

Cinematic blurred fixed background with white bold text.

Simple input for user ID and instant recommendations.

Combined recommendations rendered prominently.

Optionally, expandable views of individual model predictions.

Dataset
The project uses the MovieLens 1M dataset:

Released in 2003, this dataset contains ~1 million ratings from ~6000 users on ~4000 movies.

The dataset includes user demographic info, movie titles, and genres.

It is ideal for evaluating and benchmarking recommendation algorithms.

Setup and Requirements
System Requirements:
Python 3.6 or later

Jupyter Notebook (for exploration and model development)

Streamlit (for app UI)

Python Dependencies:
Install latest versions of the following:

pandas

numpy

scipy

scikit-learn

torch (PyTorch)

streamlit

Example installation:

bash
pip install pandas numpy scipy scikit-learn torch streamlit
How to Use
Launch the Streamlit app:

bash
streamlit run app/streamlit_app.py
Enter a valid User ID from the dataset.

Click “Get Recommendations” to see hybrid movie suggestions.

Optionally explore individual recommendation models.

Future Work
Add onboarding flows for new users to rate sample movies and gather preferences.

Integrate Bollywood-specific movie datasets and metadata.

Implement user login and persistent profiles.

Incorporate richer movie details, reviews, and interactivity.

License
This project is released under the MIT License. See LICENSE file for details.

