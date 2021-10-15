 Load libraries 
import json
import os, pickle
import pandas as pd 
import numpy as np
from urllib import request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from azureml.core import model

def init():
    # Create a global variable for loading the model
    global model
    model = MLPRegressor(hidden_layer_sizes=(512,1024,1024,512 ))
    with open(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "finalized_model.sav"), 'rb') as file:
        model=pickle.load(file)

def run(request):
    # Receive the data and run model to get predictions 
    data = pd.read_json(request)
    X = get_X(data)
    y = model.predict(X)
    print(y)
    return json.dumps(y.tolist())

def get_X(clean_data):
    vectorizer = TfidfVectorizer(lowercase=True,token_pattern=r'(?u)\b[A-Za-z]+\b',stop_words='english',max_features=2000,strip_accents='unicode')
    X=vectorizer.fit_transform(clean_data['excerpt'].values)

    return X
