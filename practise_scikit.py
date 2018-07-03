# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:44:42 2018

@author: ilu-pc
"""

import pandas as pd

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
# Create and fit a nearest-neighbor classifier
from sklearn import neighbors

#load dataset
iris = datasets.load_iris()
print (iris.data.shape)
#data converted into dataframe
df = pd.DataFrame(iris.data,columns = [iris.feature_names])
df['target'] = pd.Series(iris.target)
X = df.iloc[:,0:4]
Y = df.iloc[:,-1] 
# training set and test set 
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=0)

#clf = svm.LinearSVC()
clf = neighbors.KNeighborsClassifier()
#learn from dataset
clf.fit(X,Y)
#predict 
result=clf.predict([[0.1, 0.2, 0.3, 0.4]])
print(result)
