from random import *
from math import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#meta de treinamento 1e-4
#learning rate 0.01
#número de neurônios 25
numberOfPoints = 250
x1 = []
x2 = []
z = []
for i in range(0,numberOfPoints):
	x1.append(uniform(-10,10))
	x2.append(uniform(-10,10))
Z = np.array(z)
X1, X2 = np.meshgrid(x1, x2)
print(X1.shape)
for i in range(0,len(X1)):
	for j in range(len(X1)):
		z.append([sin(sqrt(X1[i][j]**2+X2[i][j]**2))/sqrt(X1[i][j]**2+X2[i][j]**2)])
z = np.array(z)
Z = z.reshape(X1.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z)
ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('Z Label')

plt.show()