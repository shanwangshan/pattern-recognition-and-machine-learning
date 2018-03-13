#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 11:30:03 2018

@author: wangshanshan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.image import imread
#task2
x=np.load('x.npy')
x=np.reshape(x,[np.size(x),1])
X=np.column_stack((x,np.ones_like(x)))
y=np.load('y.npy')
# =============================================================================
# y=np.reshape(y,[np.size(y),1])
# =============================================================================
# =============================================================================
# a,b=np.linalg.inv(X.T@X)@X.T@y
# =============================================================================
a,b= np.linalg.lstsq(X,y)[0]
print(a,b)
print('a is',a,'b is',b)
plt.figure(1)
plt.plot(x,y,'ro')
plt.plot(x,a*x+b)
plt.show()
#task3
#a)
if __name__ == "__main__":
    data=[]
    with open('locationData.csv','r') as fp:
        for line in fp:
# =============================================================================
#             print(line
# =============================================================================
            values=line.strip().split(' ')
            
            values = [float(v) for v in values] #convert string to float
            data.append(values)
        print(data[0:2])
            
print(np.shape(data))
#b)
data1=np.loadtxt('locationData.csv')
print(np.shape(data1))
#check whether the data are the same or not by using two different way to load the data
print(np.all(data==data1))
#task4
mat = loadmat("twoClassData.mat")
print(mat.keys())
X = mat["X"]
print(np.shape(X))
y = mat["y"].ravel()
print(np.shape(y))
print(y)
plt.figure(2)
plt.plot(X[y==0, 0], X[y==0, 1],'ro')
plt.plot(X[y==1, 0], X[y==1, 1],'bo')
plt.show()
#task5
plt.figure(3)
Z=imread('uneven_illumination.jpg')
plt.imshow(Z,cmap='gray')
plt.title("Image shape is %dx%d" % (Z.shape[1], Z.shape[0]))
X, Y = np.meshgrid(range(1300), range(1030))
x=X.ravel()
y=Y.ravel()
z=Z.ravel()
print(z)
ones=np.ones_like(x)
H = np.column_stack([x*x,y*y,x*y,x,y,ones])
c= np.linalg.lstsq(H,z)[0]
z_pred = np.dot(H, c)
z=z-z_pred 
z=np.resize(z,np.shape(Z))
plt.figure(4)
plt.imshow(z,cmap='gray')