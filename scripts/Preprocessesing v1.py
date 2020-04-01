# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:45:12 2019

@author: billy
"""

# Pre-processing

import os
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
import unicodedata
import en_core_web_sm


# Import
os.chdir("D:/Research/Echo NLP Project/")

df = pd.read_csv("Cleaned_Corpora.txt", sep = "|")

# Using the latest English Core Model Data v1.2.1 release 2017-03-21
nlp = en_core_web_sm.load(parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()

#Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#Expanding Contractions
#Contractions are shortened version of words or syllables.
exec(open("contractions.py").read())    

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#Removing Special Characters
# Special characters and symbols are usually non-alphanumeric characters or even occasionally numeric characters (depending on the problem), which add to the extra noise in unstructured text   
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

# Stemming
#Word stems are also known as the base form of a word, and we can create new words by attaching affixes to them in a process known as inflection
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# vs snowball stemmer
def snowball_stemmer(text):
    ss = nltk.SnowballStemmer(language='english')
    text = ' '.join([ss.stem(word) for word in text.split()])
    return text
  
#Lemmatisation
# Remove word affixes to get to the base form of a word
# Nb: Ensure the tagger is disabled in NLP to preserver captial letters
#     rule-based lemmatizer with re-casing can be too aggressive for our needs - as we do want proper nouns to retain their case.
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_.lower() if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


#Removing Stopwords
#Words which have little or no significance, especially when constructing meaningful features from text, are known as stopwords or stop words.
#Nb: is_lower_case set to false because many stopwords occur at the start of a sentence (which is capitalised).
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Final Text Normaliser
#    nb: lowercase removal not useful for NER of organisations
def normalize_corpus(corpus, 
                     contraction_expansion=True,
                     accented_char_removal=True,
                     text_lemmatization=True, 
                     special_char_removal=True, 
                     stopword_removal=True, 
                     remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # remove duplicate spaces (double or more spacing)
        doc = " ".join(doc.split())
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

# pre-process text and store the same
df['fulltext'] = normalize_corpus(df['text'])

example = ["1. Normal LV size with moderate to severe systolic impairment. 2. Moderate RV impairment. Note follow up echo would best with echo contrast. All LV segments are abnormal to some degree. The apex is mildly expanded. Overall there is severe LV systolic impairment with estimated LVEF ~25 - 30%. Unable to exclude LV thrombus as the apex is not well seen."]

normalize_corpus(example)

print(df['fulltext'][1])

# Export as csv (for R)
df.to_csv("Research Report/Clean_data_v1.csv")

# Export as python data frame
df.to_pickle("Research Report/Clean_data_v1.pkl")







