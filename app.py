#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import joblib
import cloudpickle
import numpy as np


# In[5]:


def word_divide_character(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters


# In[6]:


# load the model from disk
# clf = pickle.load(open('nlp_model.pkl', 'rb'))
# cv = pickle.load(open('tranform.pkl','rb'))

# transform_load_model = joblib.load('transformmmmmm.sav')
# nlp_load_model = joblib.load('nlp_model.sav')

file1 = open('transform.pkl','rb')
transform_load_model = pickle.load(file1)
file1.close()
file2 = open('nlp_model1.pkl','rb')
nlp_load_model= pickle.load(file2)
file2.close()

app = Flask(__name__)


# In[7]:


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    passs = str(request.form['password'])
    data = np.array([passs])
    
    vect = transform_load_model.transform(data)

    my_prediction = nlp_load_model.predict(vect)
    
    if my_prediction[0] == 0:
        msg = 'WEAK PASSWORD!!!'
    elif my_prediction[0] == 1:
        msg = 'NORMAL PASSWORD!!'
    else:
        msg = 'STRONG PASSWORD!!' 
    return render_template('index.html',prediction = msg)
   

if __name__ == '__main__':
	app.run(debug=True)
    
    
