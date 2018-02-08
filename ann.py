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

# Importing the dataset
dataset = pd.read_csv('/home/amansawhney/Development/language-classifier/lang_data.csv')
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(y)

X = []
j = 0
for i in x:
    if not (isinstance(i, str)):
        print(i)
        print(j)
        x = np.delete(x, j)
        y = np.delete(y, j)
    else:
        j += 1
        X.append(list(i))

labelencoder_X = LabelEncoder()
X = labelencoder_X.fit_transform(X)

X = np.asarray(X)
X = X.reshape(-1,1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
