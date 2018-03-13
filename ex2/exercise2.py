
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
from scipy.io import loadmat
from matplotlib.image import imread

# task2
x=np.load('x.npy')
y=np.load('y.npy')
# print(x)
N=len(x)
b=(np.sum(y)*np.sum(x*x)-np.sum(x*y)*np.sum(x))/(N*np.sum(x*x)-(np.sum(x))*(np.sum(x)))
a=(np.sum(x*y)-b*np.sum(x))/np.sum(x*x)
print('The value of a is', a)
print('The value of b is', b)

# task3
# a)
if __name__ == "__main__":
    # data = []
    with open("locationData.csv", "r") as fp:
        data = []
        for line in fp:
            data1 = line.strip().split(" ")
            data1 = [float(v) for v in data1]
            data.append(data1)
        print(data)
        print(np.shape(data
                       ))   #how does it work?
# b)
if __name__ == "__main__":
    data1 = np.loadtxt('locationData.csv')


    print(np.all(data1==data))
    # data = np.any(data)

# task4
# a)
mat=loadmat('twoClassData.mat')
print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()  # why ravel?
print(X)
print(y)
# b)
plt.figure(1)
X[y == 0, :]
plt.plot(X[y==0, 0], X[y==0, 1], 'ro')
X[y == 1, :]
plt.plot(X[y==1, 0], X[y==1, 1], 'bo')
plt.show()
# task5
plt.figure(2)
img = imread("uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))

plt.show()
# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img
x = X.ravel() #flatten the vector
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
theta = np.linalg.lstsq(H,z)[0]  # [0]means the first element of array
theta = np.matrix(theta)
# Predict
z_pred = np.matrix(H) @ theta.T #T means transpose, @ means matrix multiplication
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.figure(3)
plt.imshow(S, cmap = 'gray')
plt.show()