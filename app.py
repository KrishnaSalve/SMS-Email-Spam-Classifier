import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")



def transform_text(input_sms):
    text = input_sms.lower()
    text = nltk.word_tokenize(text)

    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing Stopwords and Punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in punctuation:
            y.append(i)

    # Stemming 
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)



if st.button("Predict"):
    # Preprocessing 
    transform_sms = transform_text(input_sms)

    # Vectorization
    vector_input = tfidf.transform([transform_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
