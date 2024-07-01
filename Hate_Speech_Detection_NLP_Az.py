# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 12:21:07 2024


"""


import pandas as pd
import numpy as np
import nltk
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download of the necessary NLTK data
nltk.download('stopwords')
stop_words = stopwords.words('english')
stemmer = PorterStemmer()


# Importing the dataset
data = pd.read_csv("C:/data/hatespeech_labeled_data.csv")



# Mapping the values for hate speech
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"})

#print(data.head(20))



# Selecting relevant columns
data = data[["tweet", "labels"]]



# Cleaning the data
def clean(text):
    text = str(text).lower()    
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stop_words]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


data["tweet"] = data["tweet"].apply(clean)



# Splitting data into features and labels
x = np.array(data["tweet"])
y = np.array(data["labels"])


# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(x)


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47)


# Model training
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# Predicting the labels for the test set
y_pred = clf.predict(X_test)



# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy of the Decision Tree Classifier:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



# Validating using a sample
sample = "kill"
sample_data = cv.transform([sample]).toarray()
print("Prediction for sample text:", clf.predict(sample_data))










