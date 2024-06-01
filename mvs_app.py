import streamlit as st
import numpy as np
import pandas as pd
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
movie_ids = []

def recommender(movie):
    movie_index = movies_list[movies_list['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(similarity[movie_index])),reverse = True,key = lambda x:x[1])[1:6]
    for i in movie_list:
        movie_ids.append(movies_list.iloc[i[0]].movie_id)
        recommended_list.append(movies_list.iloc[i[0]].title)
    return recommended_list,movie_ids

if st.button('Recommend'):
    recommended_movies,recommended_ids = recommender(selected_movie)
    for i,j in recommended_movies,recommended_ids:
        st.write(i,j)
