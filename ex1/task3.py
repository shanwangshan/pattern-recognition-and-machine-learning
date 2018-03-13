#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:28:08 2018

@author: wangshanshan
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
a=np.loadtxt('locationData.csv')

def normalize_data(X):
    m=np.mean(X)
    s=np.std(X)
    X_norm=(X-m)/s
    return [m, s, X_norm]
#print(normalize_data(a[:,0]))
X_norm1 = normalize_data(a[:,0])[2]
y1=normalize_data(X_norm1)[0:2]
print(y1)
X_norm2 = normalize_data(a[:,1])[2]
y2=normalize_data(X_norm2)[0:2]
print(y2)

X_norm3 = normalize_data(a[:,2])[2]
y3=normalize_data(X_norm3)[0:2]
print(y3)
#print(normalize_data(a[:,1]))
#print(normalize_data(a[:,2]))
    
