import streamlit as st
import numpy as np
import pandas as pd
import requests
import ast
import nltk
from nltk.stem.porter import PorterStemmer

st.title('Movies Recommender')
movies_list = pd.read_csv('movies_list.csv')
selected_movie = st.selectbox('Choose Movie for Recommendation',movies_list['title'].values)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(movies_list['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

recommended_list = []
movie_posters = []

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=e-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

def recommender(movie):
    movie_index = movies_list[movies_list['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(similarity[movie_index])),reverse = True,key = lambda x:x[1])[1:6]
    for i in movie_list:
        movie_posters.append(fetch_poster(movies_list.iloc[i[0]].movie_id))
        recommended_list.append(movies_list.iloc[i[0]].title)
    return recommended_list,movie_posters

if st.button('Recommend'):
    recommended_movies,recommended_posters = recommender(selected_movie)
    col1, col2, col3,col4,col5 = st.columns(5)
    with col1:
            st.header(recommended_movies[0])
            st.image(recommended_posters[0])
    with col2:
            st.header(recommended_movies[1])
            st.image(recommended_posters[1])
