#!/usr/bin/env python
# coding: utf-8

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[10]:


import json
from collections import Counter
from keras.models import Model
from keras.layers import Embedding, Input, Reshape
from keras.layers.merge import Dot
from sklearn.linear_model import LinearRegression
import numpy as np
import random
from sklearn import svm
with open('/kaggle/input/found_movies.ndjson') as fin:
    movies = [json.loads(l) for l in fin]


# In[11]:


# Remove non-movie articles
movies_with_wikipedia = [movie for movie in movies if 'Wikipedia:' in movie[0]]
movies = [movie for movie in movies if 'Wikipedia:' not in movie[0]]
print(f'Found {len(movies)} movies.')


# In[12]:


n = 21
movies[n][0], movies[n][1], movies[n][2][:5], movies[n][3][:5], movies[n][3][:5], movies[n][4], movies[n][5]


# In[13]:


link_counts = Counter()
for movie in movies:
    link_counts.update(movie[2])
link_counts.most_common(10)


# In[14]:


#creating mappings to integers.When we feed movies into the embedding neural network, we will have to represent them as numbers, and this mapping will let us keep track of movies.
top_links = [link for link, c in link_counts.items() if c >= 4]
link_to_idx = {link: idx for idx, link in enumerate(top_links)}
movie_to_idx = {movie[0]: idx for idx, movie in enumerate(movies)}
idx_to_movie = {v: k for k, v in movie_to_idx.items()}

movie_to_idx["Batman Begins"]


# In[15]:


import pickle


# In[16]:


#building a training set
pairs = []
for movie in movies:
    pairs.extend((link_to_idx[link], movie_to_idx[movie[0]]) for link in movie[2] if link in link_to_idx)
pairs_set = set(pairs)
len(pairs), len(top_links), len(movie_to_idx)


# In[17]:


import numpy as np
import random
random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (link_id, movie_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (link_id, movie_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            movie_id = random.randrange(len(movie_to_idx))
            link_id = random.randrange(len(top_links))
            
            # Check to make sure this is not a positive example
            if (link_id, movie_id) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (link_id, movie_id, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'link': batch[:, 0], 'movie': batch[:, 1]}, batch[:, 2]


# In[18]:


next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))


# In[19]:


#modeling
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model


# In[20]:


def movie_embedding_model(embedding_size = 50, classification = False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a link is present in a article"""
    
    # Both inputs are 1-dimensional
    movie = Input(name = 'movie', shape = [1])
    link = Input(name = 'link', shape = [1])
    
    # Embedding the book (shape will be (None, 1, 50))
    movie_embedding = Embedding(name = 'movie_embedding',
                               input_dim = len(movie_to_idx)+540,
                               output_dim = embedding_size)(movie)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(top_links)+540,
                               output_dim = embedding_size)(link)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [movie, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [movie, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = movie_embedding_model()
model.summary()


# In[21]:


n_positive = 1024

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(gen, epochs = 15, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)


# In[23]:


model.save("my_model")


# In[24]:


# Extract embeddings
movie_layer = model.get_layer('movie_embedding')
movie_weights = movie_layer.get_weights()[0]
movie_weights.shape


# In[25]:


movie_weights = movie_weights / np.linalg.norm(movie_weights, axis = 1).reshape((-1, 1))
movie_weights[0][:10]
np.sum(np.square(movie_weights[0]))


# In[26]:


movie_lengths = np.linalg.norm(movie_weights, axis=1)
normalized_movies = (movie_weights.T / movie_lengths).T

def similar_movies(movie):
    dists = np.dot(normalized_movies, normalized_movies[movie_to_idx[movie]])
    closest = np.argsort(dists)[-10:]
    for c in reversed(closest):
        print(c, movies[c][0], dists[c])

similar_movies('Rogue One')


# In[27]:


similar_movies('Harry Potter and the Prisoner of Azkaban (film)')


# In[30]:


similar_movies('Eega')


# In[37]:


similar_movies('K.G.F: Chapter 2')


# In[39]:


# Save the mappings
with open('link_to_idx.pickle', 'wb') as handle:
    pickle.dump(link_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('link mapping saved to disk')

with open('movie_to_idx.pickle', 'wb') as handle:
    pickle.dump(movie_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('movie mapping saved to disk')

with open('idx_to_movie.pickle', 'wb') as handle:
    pickle.dump(idx_to_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('movie reverse mapping saved to disk')


# In[38]:


pickle.dump(model,open('model.pkl','wb'))


# In[40]:


pickle.dump(normalized_movies,open('normalized_movies.pkl','wb'))


# In[42]:


pickle.dump(movie_lengths,open('movie_lengths.pkl','wb'))


# In[ ]:


pickle.dump(movie_weights,open('movie_weights.pkl','wb'))

