#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:00:29 2019

@author: mud
"""

## Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Import dataset
dataset = pd.read_csv("../Data.csv")
# matrix of features
X = dataset.iloc[:, :3].values
# Dependent variable vector
y = dataset.iloc[:, 3:].values


## Missing numeric data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


## encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# but this gives 0,1,2 values to countries which imposes an order on them
# therefore use OneHotEncoder
# both encoders are necessary, first LabelEncoder, then OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# onehotencoder creates 'dummy variables'

# encoder dependent variable matrix
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
# here, this gives values 0 & 1 which could be used as boolean


## train - test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# on above line, don't need to provide train_size because, it is (1 - test_size)
# "a good number for random_state is 42"


## feature-scaling
# q) What is the need to do feature-scaling?
# ans)  ml models are based on euclidean distance, so orders of magnitude matter a lot
#       here, age is O(10) and salary is O(10000)

# q) What are types of feature-scaling?
# ans) 1. Standardization i.e. minus mean, div sd
#       2. normalization ie. minus minimum, div range
# and MODEL also converges FASTER e.g. in decision trees

# q) should we scale dummy variables?
# ans) depends on context

# how to?
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # fit_transform on train set
X_test = sc_X.transform(X_test) # just xform on test set



