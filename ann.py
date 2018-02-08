#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:42:49 2018

@author: amansawhney
"""

# Classification template

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# Importing the dataset
dataset = pd.read_csv('/home/amansawhney/Development/language-classifier/lang_data.csv')
dataset = dataset.sample(10000).fillna(value = 0)
x = dataset.iloc[:, 2:].values
labelencoder_X = LabelEncoder()
X = labelencoder_X.fit_transform(x)

y = dataset.iloc[:, 1].values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#Init ANN
classifier = Sequential()

#add the input layers and the first hidden layer
classifier.add(Dense(output_dim = 3, init='uniform', activation='relu', input_dim= 1))
classifier.add(Dense(output_dim = 3, init='uniform', activation='relu'))


#add output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss= 'categorical_crossentropy', metrics= ['accuracy'])

# Fitting classif. 1ier to the Training set
classifier.fit(X_train, y_train, batch_size = 1000, nb_epoch=15)

score = classifier.evaluate(X_test, y_test, batch_size=128)





