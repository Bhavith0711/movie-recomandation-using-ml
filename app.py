import streamlit as st
from recommender import movies, content_recommendations, collaborative_recommendations

st.set_page_config(page_title="Movie Recommendation System", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie recommendations using AI/ML.")

option = st.radio("Choose recommendation type:", ["Content-Based", "Collaborative Filtering"])

if option == "Content-Based":
    movie = st.selectbox("Select a movie:", movies["title"].values)
    if st.button("Recommend"):
        st.write("### Recommended Movies:")
        recs = content_recommendations(movie, 5)
        for r in recs:
            st.write(f"- {r}")

else:
    user_id = st.number_input("Enter user ID (1–943):", min_value=1, max_value=943, step=1)
    if st.button("Recommend"):
        st.write("### Recommended Movies:")
        recs = collaborative_recommendations(user_id, 5)
        for r in recs:
            st.write(f"- {r}")
