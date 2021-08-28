import os
import pickle
import streamlit as st
import sys
import nltk  # Text libarary
# nltk.download('stopwords')
# nltk.download('wordnet')
import string  # Removing special characters {#, @, ...}
import re  # Regex Package
from nltk.corpus import stopwords  # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer  # Stemmer & Lemmatizer

model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
model = pickle.load(open(model_name, 'rb'))
vectorizer = pickle.load(open(vectorizer_name, 'rb'))
punc = string.punctuation
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def clean(review):
    review = ' '.join([word for word in review.split() if word not in punc])
    review = ' '.join([word for word in review.split()
                       if word not in (stop_words)])
    review = ' '.join([stemmer.stem(word) for word in review.split()])
    review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    return review


def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = clean(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"


def main():
    st.markdown("<h1 style='text-align: center; color: #f0883e;background-color:#0E1117'>Food Review Classifier</h1>", unsafe_allow_html=True)
    review = st.text_input(label='Write The Review')
    if st.button('Classify'):
        result = raw_test(review, model, vectorizer)
        st.success(
            'This Review Is {}'.format(result))


if __name__ == '__main__':
    main()
