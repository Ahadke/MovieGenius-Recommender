# **MovieGenius: Hybrid Movie Recommender System**

**MovieGenius** is a hybrid movie recommendation system leveraging content-based filtering, collaborative filtering, matrix factorization, and neural networks on the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). Built with Python and Streamlit, this project offers personalized, interactive movie suggestions through a clean and cinematic user interface.

---

## **Project Overview**

This system combines four recommendation methods to provide comprehensive movie suggestions:

- **Content-Based Filtering:** Recommends movies similar to the user's preferences based on genres and features.
- **Collaborative Filtering:** Uses user-user similarities to predict individual tastes.
- **Matrix Factorization:** Models latent factors for accurate predictions.
- **Neural Network Model:** Captures complex user-item relationships to refine recommendations.

The app features a cinematic blurred background with a fixed image, stylish white bold text, and real-time recommendations.

---

## **Dataset**

The project uses the renowned [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/), containing over one million ratings from more than 6,000 users on nearly 4,000 movies, widely utilized for recommender system benchmarking.

---

## **Requirements**

- Python 3.6 or higher
- Streamlit
- PyTorch
- pandas, numpy, scikit-learn and other ML libraries

Install dependencies via:
pip install -r requirements.txt


---

## **Usage**

Run the app locally with:
streamlit run app/streamlit_app.py

Enter a valid user ID from the dataset to receive personalized hybrid movie recommendations.

---

## **Future Enhancements**

- Onboarding flows enabling new users to rate movies and select genres.
- Integration of Bollywood-specific movie datasets.
- User profiles with demographic details.
- Rich movie metadata including reviews, trailers, and photos.

---

## **License**

This project is open source under the **MIT License**. See the LICENSE file for details.

---

*This README provides an accessible overview to users and contributors, clearly communicating the project's purpose, usage, and roadmap.*

