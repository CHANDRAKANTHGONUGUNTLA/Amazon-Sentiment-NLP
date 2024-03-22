# Sentiment Analysis on Amazon Fire Stick Reviews
### Introduction
This project aims to perform sentiment analysis on Amazon product reviews using natural language processing (NLP) techniques. The sentiment analysis is conducted to classify the sentiment of the reviews into positive, negative, or neutral categories based on the ratings provided by users.

### Importing Necessary Libraries
```python
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from nltk.tokenize import word_tokenize
from cleantext import clean
from wordcloud import WordCloud, STOPWORDS
```
This section imports all the necessary libraries and modules required for data preprocessing, natural language processing, machine learning, and visualization.

### Downloading NLTK Dependencies
```python
for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset",
    "stopwords"
):
    nltk.download(dependency)
```
This part downloads essential NLTK corpora and resources needed for text preprocessing and analysis.

### Suppressing Warnings and Seeding Random State
```python
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)
```
Here, warnings are suppressed to avoid cluttering the output, and the random seed is set to ensure reproducibility of results.

### Loading and Preprocessing Data
```python
reviews = pd.read_csv('Fire Stick.csv')
reviews.dropna(axis=0, subset=['text'], inplace=True)
reviews = reviews.drop_duplicates(subset=['id'], keep='first')
```
This section loads data from a CSV file, drops rows with missing text data, and removes duplicate entries based on the 'id' column.

### Defining Label Function and Applying Labels
```python
def label(value):
    if value > 3:
        return 1
    elif value < 3:
        return -1
    else:
        return 0

reviews['label'] = reviews['rating'].apply(label)
reviews.drop(columns=['id', 'profileName', 'date', 'title', 'images', 'Configuration', 'helpful'], axis=1, inplace=True)
```
The label function assigns sentiment labels based on the 'rating' column, and these labels are applied to the dataset.

### Text Preprocessing Function
```python
def text_preprocessor(comment):
    # Removing http links
    comment = re.sub(r"http\S+", " ", comment)
    # Removing special tags
    comment = BeautifulSoup(comment, 'lxml').get_text()
    # Decontracting the words
    comment = re.sub(r"won't", "will not", comment)
    comment = re.sub(r"can\'t", "can not", comment)
    # General contractions
    comment = re.sub(r"n\'t", " not", comment)
    comment = re.sub(r"\'re", " are", comment)
    comment = re.sub(r"\'s", " is", comment)
    comment = re.sub(r"\'d", " would", comment)
    comment = re.sub(r"\'ll", " will", comment)
    comment = re.sub(r"\'t", " not", comment)
    comment = re.sub(r"\'ve", " have", comment)
    comment = re.sub(r"\'m", " am", comment)
    # Removing words with numbers
    comment = re.sub("\S*\d\S*", ' ', comment)
    # Removing special characters
    comment = re.sub("[^A-Za-z0-9]+", ' ', comment)
    # Removing numbers
    comment = re.sub("\d+", ' ', comment)
    # Removing stopwords & stemming
    return ' '.join(stemmer.stem(word.lower()) for word in comment.split() if word not in stopwords.words('english'))
```
This function preprocesses text data by performing tasks like removing URLs, HTML tags, contractions, special characters, numbers, and stopwords. It also applies stemming to reduce words to their root form.

### Applying Text Preprocessing to the Data
```python
reviews["clean_message"] = reviews["text"].apply(text_preprocessor)
```
The text_preprocessor function is applied to the 'text' column of the dataset, and the preprocessed text is stored in a new column called 'clean_message'.

### Splitting Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(
    reviews["clean_message"],
    reviews["label"],
    test_size=0.25,
    random_state=43,
    shuffle=True,
    stratify=reviews["label"],
)
```
The dataset is split into training and testing sets, with 75% of the data used for training and 25% for testing. The data is shuffled, and the stratify parameter ensures that the class distribution is maintained in the splits.

### Vectorization:
```python
vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(X_train)
X_train_trans = vectorizer.transform(X_train)
X_text_trans = vectorizer.transform(X_test)
```
Here, the code initializes a CountVectorizer object to convert text data into numerical features. It fits the vectorizer on the training data to build the vocabulary and then transforms both the training and testing data into sparse matrices of token counts (X_train_trans and X_text_trans, respectively).

### Multinomial Naive Bayes Classifier:
```python
spam_classifier = MultinomialNB()
```
A Multinomial Naive Bayes classifier is instantiated to be used for sentiment analysis. Naive Bayes classifiers are commonly used for text classification tasks due to their simplicity and efficiency.

### Cross-Validation:
```python
scores = cross_val_score(spam_classifier, X_train_trans, y_train, cv=10, verbose=3, n_jobs=-1)
```
This part applies 10-fold cross-validation on the training data to evaluate the performance of the Multinomial Naive Bayes classifier. Cross-validation helps to estimate how well the model will generalize to unseen data.

### Model Training and Evaluation:
```python
spam_classifier.fit(X_train_trans, y_train)
y_pred = spam_classifier.predict(X_text_trans)
accuracy_score(y_test, y_pred)
```
The Multinomial Naive Bayes classifier is trained on the training data, and then predictions are made on the test data. The accuracy of the model is computed by comparing the predicted labels (y_pred) with the true labels (y_test).
These sections focus on building and evaluating a sentiment analysis model using the Multinomial Naive Bayes classifier. The model is trained on the preprocessed text data and evaluated using cross-validation and accuracy metrics. Additionally, the dataset is split into training and testing sets to assess the model's performance on unseen data.

### Model Fine-Tuning:
```python
distribution = {"alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0, 0.2, 0.3]}
grid = RandomizedSearchCV(
    spam_classifier,
    param_distributions=distribution,
    n_jobs=-1,
    cv=10,
    n_iter=20,
    random_state=42,
    return_train_score=True,
    verbose=2,
)
grid.fit(X_train_trans, y_train)
```
This section involves fine-tuning the hyperparameters of the Multinomial Naive Bayes classifier using RandomizedSearchCV. It defines a grid of alpha values and performs a randomized search over the hyperparameter space to find the best combination of parameters that optimize model performance.

### Best Model Summary:
```python
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
```
After the randomized search, this code snippet prints out the best cross-validated score, the best estimator (model), and the best parameters found during the search process. This information helps in understanding which combination of hyperparameters produces the most optimal model performance.

### Model Saving:
```python
joblib.dump(spam_classifier, 'emotion-detection-model.pkl')
```
Finally, the trained Multinomial Naive Bayes classifier is saved to a file named 'emotion-detection-model.pkl' using the joblib library. This allows the model to be reused later for making predictions on new data without needing to retrain it from scratch.

### Pipeline Creation:
```python

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_preprocessor)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier()),
])
```
Here, a machine learning pipeline is created using scikit-learn's Pipeline class. The pipeline consists of three steps: text preprocessing using CountVectorizer, TF-IDF transformation using TfidfTransformer, and classification using a RandomForestClassifier.

### Pipeline Training and Evaluation:
```python
pipeline.fit(X_train, y_train)
y_preds = pipeline.predict(X_test)
accuracy_score(y_test, y_preds)
```
The pipeline is trained on the preprocessed training data (X_train and y_train) and then used to make predictions on the test data (X_test). The accuracy of the pipeline's predictions is calculated and printed out to assess its performance.

These sections extend the model building process by fine-tuning hyperparameters, saving the trained model, and creating a pipeline for preprocessing and classification. The pipeline encapsulates the entire workflow, making it easy to apply the same preprocessing steps and model to new data.


### Alternative Classification Models:
```python
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = [MultinomialNB(alpha=1), DecisionTreeClassifier(), ExtraTreeClassifier(), LogisticRegression(), RandomForestClassifier(), SVC()]

def check(model):
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_preprocessor)),
        ('tfidf', TfidfTransformer()),
        ('classifier', model),
    ])
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)
    return accuracy_score(y_test, y_preds)

acc_scores = []
for model in models:
    res = check(model)
    print(model, res)
```
This section explores the performance of various classification models other than Multinomial Naive Bayes. It imports several classifiers from scikit-learn and evaluates their accuracy using the same preprocessing pipeline as before. The check function iterates over each model, trains it on the training data, makes predictions on the test data, and computes the accuracy score. Finally, the accuracy scores of all models are printed for comparison.

### Text Vectorization with n-grams:
```python
bow = CountVectorizer(analyzer=text_preprocessor, ngram_range=(2, 3)).fit(reviews['text'])
bow_transformer = bow.transform(reviews['text'])
transformer = TfidfTransformer().fit(bow_transformer)
tfidf = transformer.transform(bow_transformer)
X = tfidf
y = reviews['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```
This part introduces the concept of n-grams for text vectorization, specifically using bi-grams and tri-grams. It creates a CountVectorizer with the ngram_range parameter set to (2, 3), which specifies that both bi-grams and tri-grams should be considered during vectorization. The text data is transformed into a TF-IDF matrix using these n-grams, and the resulting features are used to train and test classification models.

### Model Training with Vectorized Text:
```python
classifier = RandomForestClassifier().fit(X_train, y_train)
pred_y_train = classifier.predict(X_train)
print(accuracy_score(y_train, pred_y_train))
```
Here, a RandomForestClassifier is trained on the vectorized text data (X_train) and corresponding labels (y_train). The trained model is then used to make predictions on the training data, and the accuracy of these predictions is computed and printed.

### Model Evaluation on Test Data:
```python
pred_y_test = classifier.predict(X_test)
print(accuracy_score(y_test, pred_y_test))
```
Similarly, the trained RandomForestClassifier is used to make predictions on the test data (X_test), and the accuracy of these predictions is calculated and printed. This provides an evaluation of how well the model generalizes to unseen data.

### Saving Vectorizer and Model:
```python
joblib.dump(bow, 'count_vectorizer.pkl')
joblib.dump(classifier, 'model.pkl')
```
Finally, both the CountVectorizer (bow) and the trained RandomForestClassifier (classifier) are saved to separate files using the joblib library. These saved objects can be loaded later for making predictions on new text data without needing to retrain the model or recompute the vectorization process.
