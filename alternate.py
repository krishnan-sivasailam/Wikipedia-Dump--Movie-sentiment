
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle
import requests
import ndjson

with open('found_movies.ndjson') as fin:
    movies = [ndjson.loads(l) for l in fin]

# Remove non-movie articles
movies_with_wikipedia = [movie for movie in movies if 'Wikipedia:' in movie[0]]
movies = [movie for movie in movies if 'Wikipedia:' not in movie[0]]
