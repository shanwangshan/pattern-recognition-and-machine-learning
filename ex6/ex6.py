#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 17:50:46 2018

@author: wangshanshan
"""
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from keras.utils import to_categorical
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from matplotlib.image import imread
#task3
x=[]
y=[]
for i in range(450):
    xtemp=imread('GTSRB_subset_2/class1/' + str(i).zfill(3) + '.jpg')
    xtemp=(xtemp-np.min(xtemp))/np.max(xtemp)
   # xtemp = np.ravel(xtemp)
    x.append(xtemp)
    y.append(0)
for j in range(210):
    xtemp=imread('GTSRB_subset_2/class2/' + str(j).zfill(3) + '.jpg')
    xtemp=(xtemp-np.min(xtemp))/np.max(xtemp)
   # xtemp = np.ravel(xtemp)
    x.append(xtemp)
    y.append(1)
x = np.array(x)    
y = np.array(y)
X_train, X_test, y_train, y_test=train_test_split(\
x, y, test_size=0.20)
# =============================================================================
# clf=LinearSVC()
# clf.fit(X_train,y_train)
# pre=clf.predict(X_test)
# acc=accuracy_score(pre, y_test)
# print(acc)
# =============================================================================
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
num_featmaps = 32 
num_classes = 2 
num_epochs = 2 
w, h = 5, 5
model = Sequential()
model.add(Conv2D(num_featmaps, (w, h), input_shape=(64, 64, 3),
activation = 'relu'))
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(28, activation = 'relu')) # Layer 4: Last layer producing 10 outputs.
model.add(Dense(num_classes, activation='softmax'))
# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='adadelta',
metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = num_epochs, validation_data = (X_test, y_test))

#task 4
N = 32 # Number of feature maps 
w, h = 5, 5 # Conv. window size
model = Sequential()
model.add(Conv2D(N, (w, h),
          input_shape=(64, 64, 3),
          activation = 'relu',
          padding = 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(N, (w, h),
          activation = 'relu',
          padding = 'same'))
model.add(MaxPooling2D((4,4)))
model.add(Flatten())
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dense(2, activation = 'sigmoid'))
print(model.summary())
#task 5

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
(X_train, y_train), (X_test, y_test) =mnist.load_data()
# Keras assumes 4D input, but MNIST is # -> Add a dummy dimension at the end.
# We use the handwritten digit database "MNIST".
X_train = X_train[..., np.newaxis] / 255.0 
X_test = X_test[..., np.newaxis] / 255.0
# Output has to be one-hot-encoded
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
num_featmaps = 32 
num_classes = 10 
num_epochs = 20 
w, h = 5, 5
model = Sequential()
# Layer 1: needs input_shape as well.
model.add(Conv2D(num_featmaps, (w, h), input_shape=(28, 28, 1),
activation = 'relu'))
# Layer 2:
model.add(Conv2D(num_featmaps, (w, h), activation = 'relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
# Layer 3: dense layer with 128 nodes
# Flatten() vectorizes the data:
# 32x10x10 -> 3200
# (10x10 instead of 14x14 due to border effect) 
model.add(Flatten())
model.add(Dense(128, activation = 'relu')) # Layer 4: Last layer producing 10 outputs.
model.add(Dense(num_classes, activation='softmax'))
# Compile and train
model.compile(loss='categorical_crossentropy', optimizer='SGD',
metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = num_epochs,batch_size=32, validation_data = (X_test, y_test))