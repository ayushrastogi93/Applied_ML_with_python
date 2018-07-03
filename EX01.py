# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:52:22 2018

@author: ilu-pc
"""

import numpy as np
import pandas as pd 
from sklearn.datasets import load_breast_cancer
 from sklearn.neighbors import KNeighborsClassifier
cancer = load_breast_cancer()
cancer.keys()

def answer_one():
    dataset = pd.DataFrame(cancer.data, columns= [cancer.feature_names])
    ds = pd.Series(cancer.target)
    dataset['target'] = ds 
    return dataset

count = [len(dataset[dataset['target']==0]), len(dataset[dataset['target']==1])]
         
X = dataset[cancer['feature_names']]
y = dataset['target']

from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
    
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    return knn
    
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    return knn.predict(means)
    
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_train)
    
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.score(X_test,y_test)    