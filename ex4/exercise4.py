#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:16:11 2018

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
import os, os.path
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

#ttask4
X_train, X_test, y_train, y_test=train_test_split(\
x, y, test_size=0.20)
#using K-NN
from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier()
model=clf.fit(X_train, y_train)
prelabel=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(prelabel,y_test)
print('the accuracy is ',accuracy1)
#using LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
model=clf.fit(X_train, y_train)
prelabel=clf.predict(X_test)
accuracy2=accuracy_score(prelabel,y_test)
print('the accuracy is ',accuracy2)
#using SVC
from sklearn.svm import SVC
clf = SVC()
model=clf.fit(X_train, y_train)
prelabel=clf.predict(X_test)
accuracy3=accuracy_score(prelabel,y_test)
print('the accuracy is ',accuracy3)


#task 5
mu = 0
sigma = 1
x=np.linspace(-5,5)
# =============================================================================
def gaussian(x,mu,sigma):
     return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))
p = gaussian(x, mu, sigma)
plt.figure(1)
plt.plot(p)
plt.show()
def log_gaussian(x,mu,sigma):
     return np.log((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.))))
p = log_gaussian(x, mu, sigma)
plt.figure(2)
plt.plot(p)
plt.show()




