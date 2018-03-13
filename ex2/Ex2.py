#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:42:16 2018

@author: Shubo Yan
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.image import imread
# Task 2 Least squares fit
xdata = np.load("x.npy")
ydata = np.load("y.npy")
N = len(xdata)
b = (np.sum([xdata*xdata])*np.sum([ydata])-np.sum([xdata])*np.sum([xdata*ydata]))/(N*np.sum([xdata*xdata])-(np.sum([xdata])*np.sum([xdata])))
a = (np.sum([ydata])-b*N)/np.sum([xdata])
print('The value of a is', a)
print('The value of b is', b)

# Task 3
# a)
if __name__ == "__main__":
    locations = []
    with open("locationData.csv", "r") as fp:
        locations = []
        for line in fp:
            location = line.strip().split(";")
            locations.append(location)
        print(locations)
#b)
if __name__ == "__main__":
    location_data=np.loadtxt('locationData.csv')
    locations = np.all(location_data)
    print(locations)        
# Task 4
# a)

mat = loadmat("twoClassData.mat")
print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()
print(X)
print(y)
print(np.shape(y))

# b)
X[y == 0, :]
plt.plot(X[y==0, 0], X[y==0, 1], 'ro')
X[y == 1, :]
plt.plot(X[y==1, 0], X[y==1, 1], 'bo')

# Task 5
# Read the data

img = imread("uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()

# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.
ones=np.ones_like(x)
H = np.column_stack([x*x,y*y,x*y,x,y,ones])
# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.
theta = np.linalg.lstsq(H,z)[0]

theta = np.matrix(theta)
# Predict

z_pred = np.matrix(H) @ theta.T
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()
