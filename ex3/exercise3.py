#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:08:34 2018

@author: wangshanshan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
from scipy.io import loadmat
from matplotlib.image import imread
from sklearn.cross_validation import train_test_split
#task 3
#a)
n= np.arange(100)
f0=0.017
w = np.sqrt(0.25) * np.random.randn(100)
x=np.sin(2*np.pi*f0*n)+w
plt.figure(1)
plt.plot(n,x,'ro')
plt.show()
#b)
scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    n = np.arange(100)
    z = -2*np.pi*1j*f*n# <compute -2*pi*i*f*n. Imaginary unit is 1j> 
    e = np.exp(z)
    #score= np.abs(np.dot(e,x))# <compute abs of dot product of x and e> 
    score = np.abs(np.convolve(e, x, 'valid'))
    scores.append(score)
    frequencies.append(f)
# Create vector e. Assume data is in x.
    
fHat = frequencies[np.argmax(scores)]
print(fHat)
#task 4
from sklearn.datasets import load_digits 
digits = load_digits()
print(digits)
print(digits.keys())
plt.figure(2)
plt.gray()
plt.imshow(digits.images[0]) 
plt.show()
print(digits.target[0])
X_train, X_test, y_train, y_test=train_test_split(\
digits.data,digits.target, test_size=0.20)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#task5
from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier()
model=clf.fit(X_train, y_train)
prelabel=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(prelabel,y_test)
print('the accuracy is ',accuracy)














