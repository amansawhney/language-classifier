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
from keras.utils import to_categorical



# Importing the dataset
dataset = pd.read_csv('/home/amansawhney/Development/language-classifier/lang_data.csv')
dataset = dataset.fillna(value = " ")
x = dataset.iloc[:, 2:]
X = x.apply(LabelEncoder().fit_transform).values

y = dataset.iloc[:, 1].values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(y)
Y = to_categorical(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

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
classifier.add(Dense(output_dim = 40, init='uniform', activation='relu', input_dim= 51))
classifier.add(Dense(output_dim = 40, init='uniform', activation='relu'))


#add output layer
classifier.add(Dense(output_dim = 3, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer = "adam", loss= 'binary_crossentropy', metrics= ['accuracy'])

# Fitting classif. 1ier to the Training set
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch=100)

score = classifier.evaluate(X_test, y_test, batch_size=128)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




