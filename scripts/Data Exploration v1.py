# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:04:57 2019

@author: billy
"""

# 2. Exploratory Analysis

import os
import pickle
import numpy as np

os.chdir("D:/Research/Echo NLP Project/")
df = pickle.load(open("Research Report/Clean_data_v1.pkl", 'rb'))

# 1.  Data Exploration / Preparation
#Examin number of words after cleaning
print(df.head(10))
print(df['fulltext'].apply(lambda x: len(x.split(' '))).sum())

# ^ There are 70861 words in the data

# Examine comments / class pairings
def print_plot(dat, index, var, label):
    example = dat[dat.index == index][[var, label]].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tag:', example[1])
        
print_plot(df, 10, 'fulltext', 'label')
print_plot(df, 711, 'fulltext', 'label')


# Run entire "exploration.py" which contains functions required below

# number of words per sample
corpus =  list(df['fulltext'])

df['nwords'] = [len(s.split()) for s in corpus]

#Plots the sample length distribution
plot_sample_length_distribution(corpus)

# 739 samples have greater than 100 words/text elements
# 883 samples has less than 50 / text elements
# 878 have between 50 and 100 words / text elements
len(df[(df['nwords']>100)])
len(df[(df['nwords']<50)])
len(df[(df['nwords']>=50) & (df['nwords']<=100)])

# median 69 / mean 74
print(np.median(df.nwords))
print(np.mean(df.nwords))

#Plots the frequency distribution of n-grams
plot_frequency_distribution_of_ngrams(corpus,
                                      ngram_range=(1, 2),
                                      num_ngrams=20) 

ClassBar = df['label'].value_counts().plot(kind='bar', title='Number of samples per class')
ClassBar.set_xlabel('Clinician-coded Labels')
ClassBar.set_ylabel('Number of samples')

df['label'].value_counts()
df['label'].value_counts()/len(df)



# ----- Baseline Results using Standard ML methods --------

# ----  A. Defining a Baseline Model ---

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

text = df['fulltext']
label = df['label']

text_train, text_test, label_train, label_test = \
train_test_split(text, label, test_size=0.2, random_state=1000) # 80/20 split

# Use BOW model to vectorize the sentences. Use CountVectorizer for this task.
# nb: create the vocabulary using only the training data
#     CountVectorizer performs tokenization which separates the sentences into a set of tokens 
#     This tokenisation process has a number of opportunities for improving performance. (eg. adding ngrams and tagsets)
vectoriser = CountVectorizer()
vectoriser.fit(text_train)

# Using this vocabulary, create the feature vectors for each sentence of the training and testing set
X_train = vectoriser.transform(text_train)
X_test = vectoriser.transform(text_test)
# ^ 770x1495 = 770 samples, each with 1501 dimensions

# -- Determine a baseline result -- 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 1. Logistic Regression
classifier = LogisticRegression() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline LogReg: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 2. Naive Bayes
classifier = MultinomialNB() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline NB: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 3. Random Forest Ensemble
classifier = RandomForestClassifier() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline RF: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 4. MLP
classifier = MLPClassifier() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline MLP: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# Using default settings across 4 different algorithms, the NB model (73% precision) and MLP model (76% precision) performed bested "out of the box" for detecting the minority class.
# MLP had highest overall accuracy of 92%


# --- forcus on severe as binary variable --

# 1. Multiclass

text = df['fulltext']
label = df['mild']

text_train, text_test, label_train, label_test = \
train_test_split(text, label, test_size=0.2, random_state=1000) # 80/20 split

vectoriser = CountVectorizer()
vectoriser.fit(text_train)

# Using this vocabulary, create the feature vectors for each sentence of the training and testing set
X_train = vectoriser.transform(text_train)
X_test = vectoriser.transform(text_test)


# 1. LogReg
classifier = LogisticRegression() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline LogReg: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 2. Naive Bayes
classifier = MultinomialNB() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline NB: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 3. Random Forest Ensemble
classifier = RandomForestClassifier() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline RF: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))

# 4. MLP
classifier = MLPClassifier() # model
classifier = classifier.fit(X_train, label_train) # fit

test_predict = classifier.predict(X_test)
print('Baseline MLP: %s' % accuracy_score(test_predict, label_test))
print(classification_report(test_predict, label_test))



















