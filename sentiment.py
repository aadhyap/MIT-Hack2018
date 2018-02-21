from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib

import pandas as pd

#nltk.download()

'''
VADAR tool:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for 
Sentiment Analysis of Social Media Text. Eighth International Conference on 
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
'''

sentence = "I hate you."

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(sentence)
for k in sorted(ss):
    print('{0}: {1}, '.format(k, ss[k]), end='')

    
''' ************************************************** '''

data = pd.read_csv("text_emotion.csv")

#rf_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier(n_jobs=-1, verbose=1))])    

#rf_clf = rf_clf.fit(data["content"], data["sentiment"])

rf_clf = joblib.load('rf.pkl')

print("Done with fitting")

predicted = rf_clf.predict([sentence])

print(predicted)
print(rf_clf.predict_proba([sentence]))
print(rf_clf.classes_)


import numpy as np

predicted = rf_clf.predict(data["content"])
print(np.mean(predicted == data["sentiment"]))

#joblib.dump(rf_clf, 'rf.pkl') 

print("\nDone")