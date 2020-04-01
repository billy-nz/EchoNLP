# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:23:11 2019

@author: billy
"""
#Data

import os
import pickle

# Import
os.chdir("D:/Research/Echo NLP Project/")
df = pickle.load(open("Research Report/Clean_data_v1.pkl", 'rb'))

# Wrapping Keras model as a Scikit Learn Model, add to pipeline
from keras import layers
from keras.models import Sequential

epochs = 15
batch_size = 64
embedding_dim = 200
num_filters = 64
kernal_size = 5
vocab_size = 1486
maxlen = 175

# Create model
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.SpatialDropout1D(0.25))
    model.add(layers.Conv1D(num_filters, kernel_size, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D())
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu')) 
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

# wrap the model using the function you created
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model,
                        epochs=epochs, batch_size=batch_size,
                        verbose=False,
                        num_filters=num_filters, kernel_size=kernal_size, 
                        vocab_size=vocab_size, embedding_dim=embedding_dim, 
                        maxlen=maxlen)

# Features
import en_core_web_sm

nlp = en_core_web_sm.load(parse=True, tag=True, entity=True)

def spacy_tokenizer(sentence):
    mytokens = nlp(sentence)
    return mytokens


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()# 



class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}




#Split Data
from sklearn.model_selection import train_test_split

text = df['fulltext']
label = df['mild']

text_train, text_test, label_train, label_test = \
train_test_split(text, label, test_size=0.2, random_state=1000) # 80/20 split


#Create Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion

pipe = Pipeline([
    ('text_union', FeatureUnion(
        transformer_list = [
            ('entity_feature', Pipeline([
                ('entity_vect', bow_vector),
            ])),
            ('keyphrase_feature', Pipeline([
                ('keyphrase_vect', tfidf_vector),
            ])),
        ],
        transformer_weights= {
            'entity_feature': 0.6,
            'keyphrase_feature': 0.2,
        }
    )),
    ('cust_trans', )        
    ('clf', model),
])

pipe.fit(X_train, X_test)


#LDA
# train a LDA Model
from sklearn import decomposition


lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['fulltext'])
xtrain_count =  count_vect.transform(text_train)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect.fit(df['fulltext'])
xtrain_tfidf =  tfidf_vect.transform(text_train)
xvalid_tfidf =  tfidf_vect.transform(text_test)

xtrain_tfidf_array = xtrain_tfidf.toarray()
xvalid_tfidf_array = xvalid_tfidf.toarray()


X_topics = lda_model.fit_transform(xtrain_tfidf)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
import numpy as np
n_top_words = 100
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))


#Test Integrate with embedding

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['fulltext'])

xtrain_emb = tokenizer.texts_to_sequences(text_train)
xvalid_emb = tokenizer.texts_to_sequences(text_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


from keras.preprocessing.sequence import pad_sequences

# NB: Run only once, else re-tokenisation required!
X_train_emb = pad_sequences(xtrain_emb, padding='post', maxlen=175)
X_test_emb = pad_sequences(xvalid_emb, padding='post', maxlen=175)


# Combine features
xtrain_tfidf_array X_train

X_train = np.concatenate([xtrain_tfidf_array, X_train_emb], axis=1)
X_test = np.concatenate([xvalid_tfidf_array, X_test_emb], axis=1)




