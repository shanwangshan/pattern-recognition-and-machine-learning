#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:33:26 2018

@author: wangshanshan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.image import imread
from sklearn.cross_validation import train_test_split
import glob
from skimage.feature import local_binary_pattern
#task3
def log_loss(w, X, y):
    """ 
    Computes the log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """
    
    L = 0 # Accumulate loss terms here.
        
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(-y[n] * np.dot(w, X[n])))
    
    return L
    
def grad(w, X, y):
    """ 
    Computes the gradient of the log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """
        
    G = 0 # Accumulate gradient here.
    
    # Process each sample in X:
    for n in range(X.shape[0]):
        
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]     # TODO: Correct these lines
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))  # TODO: Correct these lines
        G += numerator / denominator
    
    return G

        
    # Add your code here:
        
    # 1) Load X and y.        
X=np.loadtxt('log_loss_data/X.csv',delimiter=',', usecols=range(2))            
y=np.loadtxt('log_loss_data/y.csv')
    # 2) Initialize w at w = np.array([1, -1])
w = np.array([1, -1])
    # 3) Set step_size to a small positive value.
step_size = 0.001
    # 4) Initialize empty lists for storing the path and
    # accuracies: 
W = []; 
accuracies = []
    
for iteration in range(200):

        # 5) Apply the gradient descent rule.
        w=w-step_size*grad(w, X, y)
        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        
        # 7) Compute the accuracy (already done for you)
            
        # Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
                # Threshold at 0.5 (results are 0 and 1)
        y_pred = (y_prob > 0.5).astype(int)
                # Transform [0,1] coding to [-1,1] coding
        y_pred = 2*y_pred - 1

        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        
        W.append(w)
    
    # 8) Below is a template for plotting. Feel free to 
    # rewrite if you prefer different style.
W = np.array(W)
plt.figure(figsize = [5,5])
plt.subplot(211)
plt.plot(W[:,0], W[:,1], 'ro-')
plt.xlabel('w$_0$')
plt.ylabel('w$_1$')
plt.title('Optimization path')
plt.subplot(212)
plt.plot(100.0 * np.array(accuracies), linewidth = 2)
plt.ylabel('Accuracy / %')
plt.xlabel('Iteration')
plt.tight_layout()
plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
plt.show()
#task4
x=[]
y=[]
for i in range(101):
    xtemp=imread('GTSRB/class1/' + str(i).zfill(3) + '.jpg')
    xtemp=local_binary_pattern(xtemp,8,5)
    tmp = np.histogram(xtemp,bins=range(0,256))
    x.append(tmp[0])
    y.append(0)
for i in range(101):
    xtemp=imread('GTSRB/class2/' + str(i).zfill(3) + '.jpg')
    xtemp=local_binary_pattern(xtemp,8,5)
    x.append(np.histogram(xtemp,bins=range(0,256))[0])
    y.append(1)
from sklearn import preprocessing
x=preprocessing.scale(x)
y=preprocessing.scale(y)
X_train, X_test, y_train, y_test=train_test_split(\
x, y, test_size=0.20)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf_list = [LogisticRegression(), SVC()]
clf_name = ['LR', 'SVC']
C_range=10.0**np.arange(-5,1)
scores=[]
for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
#task5
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
classifiers = [(RandomForestClassifier(n_estimators=100), "Random Forest"), (ExtraTreesClassifier(n_estimators=100), "Extra-Trees"),
(AdaBoostClassifier(n_estimators=100), "AdaBoost"), (GradientBoostingClassifier(n_estimators=100), "GB-Trees")]
accuracies1 = []
for clf, name in classifiers: 
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    accuracies1.append(accuracy)
    

 