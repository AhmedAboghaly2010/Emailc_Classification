import streamlit as st
import pickle
import pandas as pd
import nltk
import string
punc=string.punctuation
from nltk.corpus import stopwords
stop=stopwords.words("English")
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def Process(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    processed_tokens = []
    for word in tokens:
        if word not in stopwords.words("english") and word not in string.punctuation:
            processed_tokens.append(word)
    stemmed_tokens=[]
    for word in processed_tokens:
        stemmed_tokens.append(ps.stem(word))
    return " ".join(stemmed_tokens)  # إعادة القائمة المعالجة

cv=pickle.load(open('vectorize.pkl','rb'))
model=pickle.load(open('model_nltk.pkl','rb'))

st.title("Email Classifier")
massage=st.text_input('Enter Your Email')

transform_massage=Process(massage)
cv_victor=cv.transform([transform_massage])
result=model.predict(cv_victor)[0]
if result ==0:
    st.text('spam')
else:
    st.text('Not Spam')    


