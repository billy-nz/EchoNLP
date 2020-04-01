# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 03:08:31 2019

@author: billy
"""
import feather
import os
import pickle
import numpy as np
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


os.chdir("D:/Research/Mortality Coding/")
df = feather.read_dataframe("1_preprocessed_sample.feather")

text = df['fulltext']
label = df['label']

# Split
text_train, text_test, label_train, label_test = \
train_test_split(text, label, test_size=0.2, random_state=1000) # 80/20 split

# TF-IDF - Word Level
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect.fit(df['fulltext'])

X_train_tfidf =  tfidf_vect.transform(text_train).toarray()
X_test_tfidf =  tfidf_vect.transform(text_test).toarray()

# TF-IDF - ngram level
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=10000)
tfidf_vect_ngram.fit(df['fulltext'])

X_train_tfidf_ng =  tfidf_vect_ngram.transform(text_train).toarray()
X_test_tfidf_ng =  tfidf_vect_ngram.transform(text_test).toarray()

# Topic Feature using LDA 
# Begins with BOW
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['fulltext'])

# transform the training and validation data using count vectorizer object
X_train_count =  count_vect.transform(text_train)
X_test_count =  count_vect.transform(text_test)

# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=40, learning_method='online', max_iter=40)
X_train_lda = lda_model.fit_transform(X_train_count)
X_test_lda = lda_model.fit_transform(X_test_count)


# Word Embedding 
# Use information from data exploration
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)

X_train_emb = tokenizer.texts_to_sequences(text_train)
X_test_emb = tokenizer.texts_to_sequences(text_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 50 # Some samples go up to 200 words!

X_train_emb = pad_sequences(X_train_emb, padding='post', maxlen=maxlen)
X_test_emb = pad_sequences(X_test_emb, padding='post', maxlen=maxlen)

# Vanilla
X_train = X_train_emb 
X_test = X_test_emb 

# Combine TF-IDF (word level) with Word Embedding
X_train = np.concatenate([X_train_tfidf, X_train_emb], axis=1)
X_test = np.concatenate([X_test_tfidf, X_test_emb], axis=1)

# Combine TF-IDF (ngram level) with Word Embedding
X_train = np.concatenate([X_train_tfidf_ng, X_train_emb], axis=1)
X_test = np.concatenate([X_test_tfidf_ng, X_test_emb], axis=1)

# Combine Topic Feature LDA with Word Embedding
X_train = np.concatenate([X_train_lda, X_train_emb], axis=1)
X_test = np.concatenate([X_test_lda, X_test_emb], axis=1)

# Combine TF-IDF (word level) + Topic LDA + Word Embedding
X_train = np.concatenate([X_train_tfidf, X_train_emb, X_train_lda], axis=1)
X_test = np.concatenate([X_test_tfidf, X_test_emb, X_test_lda], axis=1)

# Combine TF-IDF (word level) + Topic LDA + Word Embedding
X_train = np.concatenate([X_train_tfidf_ng, X_train_emb, X_train_lda], axis=1)
X_test = np.concatenate([X_test_tfidf_ng, X_test_emb, X_test_lda], axis=1)



#----- Modeling ----
from keras.models import Sequential
from keras import layers
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score


# --- Model ---

epochs = 15
batch_size = 128
embedding_dim = 200
num_filters = 128
kernal_size = 5
input_len = X_train.shape[1]

model = Sequential()

# Layer 1. Word Embedding
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=input_len))
model.add(layers.Conv1D(num_filters, kernal_size, padding='same', activation='relu'))
model.add(layers.MaxPooling1D(4))
# Layer 2: Spatial Dropout
#model.add(layers.SpatialDropout1D(0.25))

# Layer 3: Convolution ID
model.add(layers.Conv1D(num_filters, kernal_size, padding='same',  activation='relu'))

# Layer 4: Bidirectional CuDNNLSTM
#model.add(layers.Bidirectional(layers.LSTM(num_filters, return_sequences=True)))
#model.add(layers.Conv1D(num_filters*2, kernal_size, activation='relu'))
# Layer 5: Max Pooling


model.add(layers.MaxPooling1D(4))

model.add(layers.Conv1D(num_filters, kernal_size, padding='same',  activation='relu'))

model.add(layers.Flatten())

# Layer 6: Dense Layer 
model.add(layers.Dense(100, activation='relu')) 
#
# Layer 7: Dropout
model.add(layers.Dropout(0.25))

model.add(layers.Dense(10, activation='relu')) 
#
# Layer 7: Dropout
model.add(layers.Dropout(0.25))

# Layer 8: Output Dense Layer
model.add(layers.Dense(1, activation='sigmoid')) # down sampling

# Create Model 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, label_train,
                    epochs = epochs,
                    verbose = False,
                    validation_data = (X_test, label_test),
                    batch_size = batch_size)

loss, accuracy = model.evaluate(X_train, label_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, label_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)


# --  Evaluation - FPR --
# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(label_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(label_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(label_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(label_test, yhat_classes)
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(label_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(label_test, yhat_probs)
print('ROC AUC: %f' % auc)



