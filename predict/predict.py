#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle


# In[2]:


movie_to_idx=pickle.load(open('E:/Movie project/movie_to_idx.pickle','rb'))
link_to_idx =pickle.load(open('E:/Movie project/link_to_idx.pickle','rb'))
idx_to_movie=pickle.load(open('E:/Movie project/idx_to_movie.pickle','rb'))
normalized_movies=pickle.load(open('E:/Movie project/normalized_movies.pkl','rb'))
movie_lengths=pickle.load(open('E:/Movie project/movie_lengths.pkl','rb'))
movie_weights=pickle.load(open('E:/Movie project/movie_weights.pkl','rb'))


# In[3]:


def similar_movies(movie):
    dists = np.dot(normalized_movies, normalized_movies[movie_to_idx[movie]])
    closest = np.argsort(dists)[-10:]
    for c in reversed(closest):
        print(c, movies[c][0], dists[c])


# In[ ]:




