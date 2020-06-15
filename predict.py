#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

import glob
import json
import re

from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#       print(os.path.join(dirname, filename))"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Similarity compute
import gensim

#saving and loading the model
import joblib

# In[6]:

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    global wordnet_lemmatizer
    global stop_words
    
    text = re.sub('<.*?>','',text) #removing html
    text = re.sub('https?:\/\/[^\s]+', '', text) #removing URL
    text = " ".join([wordnet_lemmatizer.lemmatize(t) for t in text.split()]) #lemmatizing
    
    #removing punctuations
    for punc in '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~':
        text = ' '.join(text.split(punc))
    
    text = ' '.join([word for word in text.split() if word not in stop_words])#removing stop_words
    
    return text.lower().strip()

# In[9]:

model = joblib.load('nlp_model.pkl') # Load "model.pkl"

#loading the files
files = []
with open('articles_body_.csv', encoding='latin-1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        files.append(row)

def getMostSimilar(question):
    question = preprocess(question) #QUESTION
    query_token = model.infer_vector(word_tokenize(question)) 
    similar_docs = model.docvecs.most_similar([query_token], topn=20) #get the top 20 similar articles to the question
    documents = [files[similar_doc[0]] for similar_doc in similar_docs] #get the data to return from the top 20 articles
    return documents

getMostSimilar("What do we know about potential risks factors?")