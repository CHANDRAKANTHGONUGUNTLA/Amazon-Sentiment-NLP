# Sentiment Analysis on Amazon Fire Stick Reviews

## Objective
Perform sentiment analysis on Amazon Fire Stick reviews to understand customer sentiments and emotions regarding the product.

## Dataset
The dataset `Fire Stick.csv` contains reviews of Amazon Fire Stick products.

## Project Flow
1. **Data Extraction and Preprocessing**
   - Load the dataset and clean the text data.
   - Perform text preprocessing tasks such as removing HTML tags, decontracting words, removing special characters, and stemming.

2. **Model Training and Evaluation**
   - Split the data into training and testing sets.
   - Vectorize text data using CountVectorizer.
   - Train a Multinomial Naive Bayes model.
   - Evaluate model performance using accuracy score.

3. **Model Deployment**
   - Save the trained model for deployment.
   - Define functions for text preprocessing and sentiment analysis.
   - Test the deployment pipeline for sentiment analysis.

## Code Overview

```python
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
reviews = pd.read_csv('Fire Stick.csv')
reviews.dropna(axis=0, subset=['text'], inplace=True)
reviews.drop_duplicates(subset=['id'], keep='first', inplace=True)

# Data preprocessing
def text_preprocessor(comment):
    # Remove HTML tags
    comment = BeautifulSoup(comment, 'lxml').get_text()
    # Decontract words
    ...
    # Remove special characters
    ...
    # Stemming
    ...
    return comment

reviews["clean_message"] = reviews["text"].apply(text_preprocessor)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    reviews["clean_message"],
    reviews["label"],
    test_size=0.25,
    random_state=43,
    shuffle=True,
    stratify=reviews["label"],
)

# Vectorize text data
vectorizer = CountVectorizer(lowercase=False)
X_train_trans = vectorizer.fit_transform(X_train)
X_test_trans = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_trans, y_train)

# Evaluate model
y_pred = nb_classifier.predict(X_test_trans)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save trained model
joblib.dump(nb_classifier, 'sentiment_analysis_model.pkl')

# Define function for sentiment analysis
def predict_sentiment(text):
    cleaned_text = text_preprocessor(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    sentiment = nb_classifier.predict(vectorized_text)[0]
    return sentiment

# Test deployment
review_text = "This Fire Stick is amazing!"
sentiment = predict_sentiment(review_text)
print("Sentiment:", sentiment)

