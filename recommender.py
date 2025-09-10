import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# ------------------------
# Load Data
# ------------------------
ratings = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
movies = pd.read_csv("data/u.item", sep="|", encoding="latin-1", header=None,
                     names=["movie_id", "title", "release_date", "video_release_date",
                            "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
                            "Children", "Comedy", "Crime", "Documentary", "Drama",
                            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                            "Romance", "Sci-Fi", "Thriller", "War", "Western"])

# ------------------------
# Content-Based Filtering
# ------------------------
genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
              "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
              "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

movies["genres"] = movies[genre_cols].idxmax(axis=1)

# TF-IDF on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def content_recommendations(title, n=5):
    """ Recommend similar movies using content-based filtering """
    if title not in indices:
        return ["Movie not found in dataset."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

# ------------------------
# Collaborative Filtering
# ------------------------
user_item_matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

nmf = NMF(n_components=20, init="random", random_state=42)
W = nmf.fit_transform(user_item_matrix)
H = nmf.components_

def collaborative_recommendations(user_id, n=5):
    """ Recommend movies for a user using collaborative filtering """
    if user_id not in user_item_matrix.index:
        return ["User not found."]
    user_idx = user_id - 1  # user IDs start from 1
    user_ratings = np.dot(W[user_idx], H)
    top_movies = user_ratings.argsort()[-n:][::-1]
    return movies[movies["movie_id"].isin(top_movies)]["title"].tolist()
