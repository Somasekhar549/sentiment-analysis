import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib as plt
import os
import re
from tqdm import tqdm
import wordninja
import unidecode
import pickle
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




    
NB_classifier = pickle.load(open("NB.pickle", "rb"))
KNN_classifier= pickle.load(open("knn.pickle", "rb"))
RF_classifier=pickle.load(open("RandomForest.pickle", "rb"))




count_vectorizer = pickle.load(open("count_vector.pickle", "rb"))
binary_count_vectorizer = pickle.load(open("binary_count_vector.pickle", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vector.pickle", "rb"))
#vectorizer=pickle.load(open("count_vector.pickel", "rb"))




st.title("Twitter Sentiment Analysis")




def clean_tweet(text):
    
    # lower-case all characters
    text=text.lower()
    
    # remove twitter handles
    text= re.sub(r'@\S+', '',text) 
    
    # remove urls
    text= re.sub(r'http\S+', '',text) 
    text= re.sub(r'pic.\S+', '',text)
      
    # replace unidecode characters
    #text=unidecode.unidecode(text)
      
    # regex only keeps characters
    text= re.sub(r"[^a-zA-Z+']", ' ',text)
    
    # keep words with length>1 only
    text=re.sub(r'\s+[a-zA-Z]\s+', ' ', text+' ') 

    # split words like 'whatisthis' to 'what is this'
    def preprocess_wordninja(sentence):      
        def split_words(x):
            x=wordninja.split(x)
            x= [word for word in x if len(word)>1]
            return x
        new_sentence=[ ' '.join(split_words(word)) for word in sentence.split() ]
        return ' '.join(new_sentence)
    
    text=preprocess_wordninja(text)
    
    # regex removes repeated spaces, strip removes leading and trailing spaces
    text= re.sub("\s[\s]+", " ",text).strip()  
    
    return text

tweet=st.text_input("Enter your tweet")
#st.session_state.tweet



def prediction(tweet):
    cleaned_text=clean_tweet(tweet)
    input=[cleaned_text]
    test_counts=count_vectorizer.transform(input)
    test_binary_counts=binary_count_vectorizer.transform(input)
    test_tfidf=tfidf_vectorizer.transform(input)
    #vect=vectorizer.transform(input)
    
    pred1 = KNN_classifier.predict_proba(test_tfidf)
    pred2=NB_classifier.predict_proba(test_counts)
    pred3=RF_classifier.predict_proba(test_binary_counts)
    #pred=classifier.predict(vect)

    pred00=np.add(pred1,pred3)
    pred01=np.add(pred00,pred2)
    #print(pred00)
    #print(pred01[:10])
    
    for i in pred01:
        if i[0]>i[1]:
            return 0
        else:
            return 1


    return pred

pred=prediction(tweet)

if len(tweet) == 0:
    st.subheader("Please enter the tweet")
elif pred:
    st.subheader("It's a negative tweet")
else:
    st.subheader("It's a positive tweet")

