from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

from collections import Counter

import pickle
import sys
import re


data = pd.read_csv('dataset.csv')

class CleanUpTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        output = list()
        for text in X:
            text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            text = re.sub(r'\W+', ' ', text, flags=re.M)
            output.append(
                ' '.join([word[:-2] for word in text.lower().split() if len(word) > 3]) # Cut off the words endings
            )
        print('[Preprocessing] Text Cleanup Completed.')
        return np.array(output)
    
class VectorizeTransformer(TransformerMixin):
    def fit(self, X, y=None, vocab_size=1000):
        vectorizer = CountVectorizer(max_features=vocab_size)
        vectorizer.fit(X)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print('[Preprocessing] Fitting Vectorizer Completed.')
        return self
    
    def transform(self, X, y=None):
        try:
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
        except FileNotFoundError:
            print('[Error] Couldn`t locate vectorizer file.')
            return np.empty([0])
            
        vector = vectorizer.transform(X)
        print('[Preprocessing] Text vectorization completed.')
        
        return vector
    
class LabelTransformer(TransformerMixin):
    def fit(self, y):
        self.encoder = LabelEncoder()
        self.encoder.fit(y.values)
        return self
    def transform(self, y):
        encoded = self.encoder.transform(y.values)
        print('[Preprocessing] Completed target labels encoding.')
        return encoded.reshape([-1, 1])
    
X_transform = Pipeline([
    ("CleanUp", CleanUpTransformer()),
    ("Vectorize", VectorizeTransformer()),
])

y_transform = Pipeline([
    ("Label", LabelTransformer())
])

X, y = data['text'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
X_train = X_train.values
X_test = X_test.values

X_train = X_transform.fit_transform(X_train).toarray()
X_test = X_transform.transform(X_test).toarray()

y_train = y_transform.fit_transform(y_train)
y_test = y_transform.transform(y_test)



model = keras.Sequential([
    layers.Dense(126, input_shape=[1000], \
                 activation='relu'),
    layers.Dense(5, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Accuracy', test_acc)
model.save('model.h5')
