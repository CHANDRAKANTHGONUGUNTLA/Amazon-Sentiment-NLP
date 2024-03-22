import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re



tf = pickle.load(open('count_vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()


def text_processing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    # text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text


st.title('Review Classification')

review_title=st.text_input("Enter the review title")

review = st.text_area('Enter the review')

review=review_title+' '+review

if st.button('Predict'):

    transformed_review = text_processing(review)

    # vectorize
    vector_ip = tf.transform([transformed_review])

    # predict
    res = model.predict(vector_ip)[0]
    m_prob = model.predict_proba(vector_ip)
    # Display

    if res == 1:
        st.success('The Review is positive' + ' ' + '(' + str(np.round(m_prob[0][1], 2)) + ')')
    else:
        st.error('The review is negative' + ' ' + '(' + str(np.round(m_prob[0][0], 2)) + ')')
