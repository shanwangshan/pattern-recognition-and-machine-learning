#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:01:55 2018

@author: wangshanshan
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
a=np.loadtxt('locationData.csv')
print(a.shape)
fig = plt.figure();
ax = fig.add_subplot(211)
ax.plot(a[:,0],a[:,1],'b--')
ax=fig.add_subplot(212, projection = "3d")
ax.plot(a[:,0],a[:,1],a[:,2])
plt.show()

