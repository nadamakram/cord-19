#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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


# In[4]:


root_path = 'E:\Machathon1.1'

#all_files = glob.glob(f'{root_path}/documnet_parses/**/*.json', recursive=True)
files_path = []
for dirname, _, filenames in os.walk(f'{root_path}\pmc_json'):
    for filename in filenames:
        files_path.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk(f'{root_path}\pdf_json'):
    for filename in filenames:
        files_path.append(os.path.join(dirname, filename))


# In[5]:


#TODO: when you finish, remove that list concatenation for running on all the data set
files_path = files_path[0:10000]


# In[6]:


def remove_html(text):
    return re.sub('<.*?>','',text)

def remove_URL(text):
    return re.sub('https?:\/\/[^\s]+', '', text)

def lemmatization(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    return " ".join([wordnet_lemmatizer.lemmatize(t) for t in text.split()]) 

def stemming(text):
    newtext =''
    lancaster=LancasterStemmer()
    for i in text.split():
        newtext += ' ' + lancaster.stem(i)
    return newtext

def remove_punc(text):
    puncs = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'
    for punc in puncs:
        text = ' '.join(text.split(punc))
    return text
def remove_stopWords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])


# In[7]:


def preprocess(text):
    text = remove_html(text)
    text = remove_URL(text)
    text = lemmatization(text)
    text = remove_punc(text)
    text = remove_stopWords(text)
    return text.lower().strip()


# In[8]:


body_sections = {}
body_sections_matters = ['abstract','introduction','discussion','conclusion','diagnosis', 'method','treatment','result','concluding','method','background','measures','transmission period','incubation']
unique_titles = []
duplicate_counter = 0

def body_sections_dic(section_name):
    global body_sections
    if section_name in body_sections: 
        body_sections[section_name] += 1
    else: 
        body_sections[section_name] = 1


def fileRead(file_path):
    with open(file_path) as file:
        content = json.load(file)
        
        body_text = []
        for entry in content['body_text']:
            preprocessed_section = preprocess(entry['section'])
            for i in body_sections_matters:
                if i in preprocessed_section or preprocessed_section == '':
                    body_text.append(entry['text'])
                    break
        newContent = preprocess('\n'.join(body_text))
        
        return newContent


# In[9]:


#TODO: save the output in a csv file when you know how to store files in kaggle or google colab
flag = 1 #if the flag is == 1, then we should read the files all over again, otherwise, it should be loaded from csv file we stored
if flag:
    files = [fileRead(eachfile) for eachfile in files_path]


# In[10]:


empty_body_cnt = 0
for indx, file in enumerate(files):
    if(file == ''):
        empty_body_cnt += 1
        files.pop(indx)
        files_path.pop(indx)
    elif file in files[:indx]:
        duplicate_counter += 1
        files.pop(indx)
        files_path.pop(indx)


# In[20]:


len(files)


# In[21]:


duplicate_counter


# In[22]:


empty_body_cnt


# In[11]:


files[0]


# In[12]:


def tagged_files(indx, file):
    tokens = word_tokenize(file)
    return gensim.models.doc2vec.TaggedDocument(tokens, [indx])


# In[13]:


Files_tokens = [tagged_files(indx, file) for indx, file in enumerate(files)]


# In[14]:


Files_tokens[:2]


# In[20]:


#TODO: Hyperparameters selection
model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2)
model.build_vocab(Files_tokens)
model.train(Files_tokens, total_examples=model.corpus_count, epochs=model.iter)


# In[21]:


#questions on task
questions = ["What do we know about potential risks factors?",
             "what is the effect of Smoking, pre-existing pulmonary disease?",
             "Do co-existing respiratory/viral infections make the virus more transmissible or virulent and other comorbidities?",
             "What is the effect on Neonates and pregnant women?",
             "What are the Socio-economic and behavioral factors on COVID-19?",
             "What is the economic impact of the virus?",
             "What are Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors?",
             "Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups",
             "What are the Susceptibility of populations?",
             "What are the Public health mitigation measures that could be effective for control?"]


# In[22]:


queries_token = [model.infer_vector(word_tokenize(q)) for q in questions]
queries_token[:2]


# In[23]:


similar_docs = [model.docvecs.most_similar([query],topn=20) for query in queries_token]
similar_docs[:2]
#Question: how to get the 8098 article, I guess it refers to articles!!!


# In[26]:


type(similar_docs)


# In[24]:


documents_found = [[files[doc[0]] for doc in question] for question in similar_docs]
documents_found


# In[25]:


import joblib
# Save the model to disk
joblib.dump(model, 'nlp_model.pkl')


# In[ ]:




