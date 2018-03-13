#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:24:40 2018

@author: wangshanshan
"""

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.io
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from keras.utils import to_categorical
import os
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from matplotlib.image import imread
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D
import  matplotlib.pyplot as plt
#task3
x=[]
y=[]
for i in range(450):
    xtemp=imread('GTSRB_subset_2/class1/' + str(i).zfill(3) + '.jpg')
    xtemp=(xtemp-np.min(xtemp))/np.max(xtemp)
    x.append(xtemp)
    y.append(0)
for j in range(210):
    xtemp=imread('GTSRB_subset_2/class2/' + str(j).zfill(3) + '.jpg')
    xtemp=(xtemp-np.min(xtemp))/np.max(xtemp)
    x.append(xtemp)
    y.append(1)
x = np.array(x)    
y = np.array(y)
X_train, X_test, y_train, y_test=train_test_split(\
x, y, test_size=0.20)
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
base_model=VGG16(include_top=False,weights='imagenet',input_shape=(64,64,3))
w=base_model.output
w=Flatten()(w)
w=Dense(100, activation='relu')(w)
output=Dense(2,activation='sigmoid')(w)
model=Model(inputs=[base_model.input], outputs=[output])
model.layers[-5].trainable=True
model.layers[-6].trainable=True
model.layers[-7].trainable=True
model.summary()
# =============================================================================
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=20,batch_size=32, validation_data=[X_test,y_test])
# =============================================================================
#taskk4

data=scipy.io.loadmat('arcene.mat')
X_train=data['X_train']
X_test=data['X_test']
y_train=data['y_train']
y_test=data['y_test']
y_train=y_train.ravel()
y_test=y_test.ravel()
rfecv = RFECV(estimator=LogisticRegression(),step=50,verbose=1) 
rfecv.fit(X_train, y_train)
# Scores and feature sets are here
plt.plot(range(0,10001,50), rfecv.grid_scores_)
numbers=rfecv.support_
pre=rfecv.predict(X_test)
acc=accuracy_score(pre,y_test)
print('accuarcy is ',acc)
#task5

clf = LogisticRegression(penalty = "l1") 
C_range =np.logspace(-6,0, 20)
accuracies = [] 
maxscore = 0 
bestScore = 0
for C in C_range: 
    clf.C = C
    score = cross_val_score(clf, X_train, y_train, cv = 10).mean()
    accuracies.append(score)  
    if(maxscore<=score):
        maxscore=score
        bestScore=C
print('maxscore', maxscore)
print('bestscore', bestScore)
clf.C=bestScore
clf.fit(X_train,y_train)
pre_y=clf.predict(X_test)
coef=clf.coef_
print(np.count_nonzero(coef))
acc=accuracy_score(y_test,pre_y)*100
print('accuracy is %.1f%%' % acc)














