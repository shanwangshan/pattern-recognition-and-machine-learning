#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:51:07 2018

@author: wangshanshan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
from scipy.io import loadmat
from matplotlib.image import imread
x=np.load('x.npy')
x=np.reshape(x,[np.size(x),1])
X=np.column_stack((x,np.ones_like(x)))
y=np.load('y.npy')
a,b= np.linalg.lstsq(X,y)[0]
print(a,b)