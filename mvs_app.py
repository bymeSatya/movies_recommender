import streamlit as st
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer

st.title('Movies Recommender')
movies_list = pd.read_csv('movies_list.csv')
option = st.selectbox('Choose Movie for Recommendation',movies_list['title'].values)
