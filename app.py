#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle
import simplejson
import requests
import json



movie_to_idx=pickle.load(open('movie_to_idx.pickle','rb'))
link_to_idx =pickle.load(open('link_to_idx.pickle','rb'))
idx_to_movie=pickle.load(open('idx_to_movie.pickle','rb'))
normalized_movies=pickle.load(open('normalized_movies.pkl','rb'))
movie_lengths=pickle.load(open('movie_lengths.pkl','rb'))
movie_weights=pickle.load(open('movie_weights.pkl','rb'))

# In[ ]:


app = Flask(__name__)


# In[ ]:


def masterlist(movie):
    dists = np.dot(normalized_movies, normalized_movies[movie_to_idx[movie]])
    closest = np.argsort(dists)[-10:]
    master=[]
    similar_movies = []
    pages=[]
    ignored=[]
    names=[]
    pageurldata=[]
    temp=[]
    newdata=[]
    mainlink=[]
    url = "https://en.wikipedia.org/w/api.php"
    URL = "https://en.wikipedia.org/w/api.php"
    S = requests.Session()
    for c in reversed(closest):
        similar_movies.append(movies[c])
    for i in range(len(similar_movies)):
        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": "File:"+str(similar_movies[i][1]['image'])}
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        PAGES = DATA["query"]["pages"]
        page = next(iter(DATA["query"]["pages"].values()))
        try:
            image_info = page["imageinfo"][0]
            image_url = image_info["url"]
            pages.append(image_url)
            ignored.append(similar_movies[i][0])
            temp.append(i)
        except KeyError:
            continue
    for i in temp[1:]:
        newdata.append(similar_movies[i])
    for i in range(len(newdata)):
        PARAMS = {
            "action": "query",
            "format": "json",
            "titles": str(newdata[i][0]),
            "prop": "info",
            "inprop": "url|talkid"}
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        PAGES = DATA["query"]["pages"]
        page = next(iter(DATA["query"]["pages"].values()))
        page_info = page["fullurl"]
        mainlink.append(page_info)
    return zip(ignored[1:],mainlink,pages[1:])


# In[ ]:

with open('found_movies.ndjson','r') as fin:
    movies = [json.loads(l) for l in fin]

# Remove non-movie articles
movies_with_wikipedia = [movie for movie in movies if 'Wikipedia:' in movie[0]]
movies = [movie for movie in movies if 'Wikipedia:' not in movie[0]]


@app.route('/')
def index():
    return render_template('index.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    # get data
    try:
        text = request.form['u']
        predict=masterlist(text)
        return render_template('index.html', Passmessage="Similar Movies for", suggestions=predict,currentmovie=text)
    except KeyError:
        return render_template('index.html',Message="Movie not in the database")


# In[ ]:


if __name__ == '__main__':
    app.run(port = 5000, debug=True)

