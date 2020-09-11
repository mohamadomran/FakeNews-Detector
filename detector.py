# Import Libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Import Data + Split into train and test sets

dataset = pd.read_csv('news.csv')
labels=dataset.label

X_train, X_test, y_train, y_test = train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 7)


# Stop Words using TfidVectorizer

vectorizer =  TfidfVectorizer(stop_words='english', max_df = 0.7)


#  Fit and Transform

tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Classification and training process

classifier = PassiveAggressiveClassifier(max_iter = 50)
classifier.fit(tfidf_train, y_train)

#Prediction + Confusion Matrix

y_prediction = classifier.predict(tfidf_test)
score = accuracy_score(y_test, y_prediction)
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test, y_prediction, labels = ['FAKE', 'REAL'])