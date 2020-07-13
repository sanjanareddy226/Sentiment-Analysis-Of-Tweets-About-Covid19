#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:49:13 2020

@author: sanjanasrinivasareddy
"""
#from sklearn.externals import joblib


import numpy as np
import os

#import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import joblib
import nltk

nltk.download('words')
words = set(nltk.corpus.words.words())
import re
from sklearn.pipeline import Pipeline
nltk.download('wordnet')        
import joblib
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
app = Flask(__name__)
model = pickle.load(open('decision.pkl', 'rb'))
sc=pickle.load(open('vector.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    
    '''
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split()]
        return clean_mess
    
    x_test = request.form.get('Tweet')
    
    
    print(x_test)
    x_test=" ".join(w for w in nltk.wordpunct_tokenize(x_test) if w.lower() in words or not w.isalpha())
    txt = x_test
    txt=re.sub(r'@[A-Z0-9a-z_:]+','',txt)#replace username-tags
    txt=re.sub(r'^[RT]+','',txt)#replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+','',txt)#replace URLs
    txt=re.sub("[^a-zA-Z]", " ",txt)#replace hashtags
    
    txt=no_user_alpha(txt)
    txt=normalization(txt)
    txt=[" ".join(txt)]
    
    x_test=sc.transform(txt)
    #print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
#    output=prediction[0][0]
#    return render_template('index.html', prediction_text='the employee stayed or no {}'.format(output))
    if(prediction==1):
        return render_template('indexx.html', prediction_text='Positive sentiment')
    else:
        return render_template('indexx.html', prediction_text='Negative sentiment')
    return render_template('index.html', prediction_text='The tweet has that following sentiment'.format(output))


    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
