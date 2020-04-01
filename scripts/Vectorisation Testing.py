# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 01:41:09 2019

@author: billy
"""





import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


os.chdir("D:/Research/Echo NLP Project/")
df = pickle.load(open("Research Report/Clean_data_v1.pkl", 'rb'))


text = df['fulltext']
label = df['mild']

# Split
text_train, text_test, label_train, label_test = \
train_test_split(text, label, test_size=0.2, random_state=1000) # 80/20 split

# Tokenize
# Use information from data exploration
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TF-IDF
import sklearn
text = []
for i in tqdm(tokens):
  string = ' '.join(i)
  text.append(string)
  
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(text_train).toarray().tolist()
X_test_vec = vectorizer.fit_transform(text_test).toarray().tolist()
tfidf_feat = vectorizer.get_feature_names()

tfidf = TfidfVectorizer() 
tfidf.fit(text_train)
tfidf_features = tfidf.transform(text_train)



# Embedding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_test = tokenizer.texts_to_sequences(text_train)

vocab_size = 175  # Adding 1 because of reserved 0 index

# Pad sequences with zeros
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# Concat
X_train2 = np.concatenate((X_train, tfidf_features), axis = 0)


#Applying TF-IDF scores to the model vectors
tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
errors=0
for sent in tqdm(tokens): # for each review/sentence
    sent_vec = np.zeros(100) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf [row, tfidf_feat.index(word)]
            sent_vec += (vec * tfidf)
            weight_sum += tfidf
        except:
            errors =+1
            pass
    sent_vec /= weight_sum
    #print(np.isnan(np.sum(sent_vec)))
tfidf_sent_vectors.append(sent_vec)
row += 1
print('errors noted: '+str(errors))





# Concat
X_train = X_train_vec + X_train_tok
X_test = X_test_vec + X_test_tok

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
#^ The indexing is ordered after the most common words in the text, which you can see by the word "the" having the index 1. 

# Examine
tokenizer.index_word # word dictionary
tokenizer.word_index['lvef'] # Look up ordered position

print(text_train[2])
print(X_train[2])
print(X_train[3])