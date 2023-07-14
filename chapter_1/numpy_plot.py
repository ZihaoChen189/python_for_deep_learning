# in python 3.x, 7/5 = 1.4; but in python 2.x, 7/5 = 1
# "python class" could be viewed as "own data type"

import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6]])  # X.shape=(3, 2)

# converted as one dimension vector
X = X.flatten()  # X.shape=(6, ) 

# print(X[X>3])  # [4 5 6] print values! not index

import matplotlib.pyplot as plt  # this could be place at the begining of the file

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin&cos')
plt.legend()
plt.show()

from matplotlib.image import imread
img = imread('xxxxxxx.png')  # not exist actually
plt.imshow(img)
plt.show()  # show this image
