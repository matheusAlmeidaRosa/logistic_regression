"""
	This program realize a logistic regression for a dataset of microchips.
	The dataset is composed of several microchips information. For each microchip is 
	know their result of two tests and if the chip is pass quality assurance or not.

	The logistic regression will try to divide the group of quality and not quality microchips.

	Author: Higor Coimbra - Matheus Rosa
	Class: Computational Inteligence
	University: CEFET-MG
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as op
import matplotlib.mlab as mlab

import sys
import math

eps = 0.000000001

def plot_data(X, y, contr,i,lamb):
	plt.figure(i)
	plt.xlabel("Mircrochip test 1")
	plt.ylabel("Mircrochip test 2")
	title = "Logistic regression with lambda = "+str(lamb)
	plt.title(title)
	c = plt.contour(contr[0], contr[1], contr[2], levels=[0.5], colors='r')

	n1 = False
	n2 = False
	for i, point in enumerate(X):
		if y[i]:
			marker = 'k+'
			label = 'Approved'
		else:
			marker = 'yo'
			label = 'Disapproved'

		if y[i] and not n1:
			plt.plot(point[0], point[1], marker, label=label)
			n1 = True
		elif not y[i] and not n2:
			plt.plot(point[0], point[1], marker, label=label)
			n2 = True
		else:
			plt.plot(point[0], point[1], marker)

	c.collections[0].set_label('Decision Boundary')
	plt.legend()

def sigmoid(x):
	if x < 0:
		return 1 - 1/(1+math.exp(x))
	else:
		return 1/(1+math.exp(-x))

def output(theta, x):
	out = 0
	for i in range(0,len(theta)):
		out += theta[i]*x[i]
	return sigmoid(out)

def cost(theta, X, y, lamb):
	m = len(X)
	summ = 1
	sum_theta = 0
	for i in range(0,m):
		h = output(theta,X[i])
		summ += (-y[i]*math.log(max(eps,h)) - (1.0-y[i])*math.log(max(eps, 1-h)))

	for th in theta:
		sum_theta += (th**2)

	return (1/m)*summ + (lamb/(2*m))*sum_theta

def gradient(theta, X, y, lamb):
	m = len(X)
	theta_new = []
	for j in range(0,len(theta)):
		summ = 0
		for i in range(0,m):
			summ += (output(theta,X[i])-y[i])*X[i][j]
		if j: 
			summ += (lamb/m)*theta[j]
		theta_new.append((1/m)*summ)
	return theta_new

def generate_contour(theta):
	
	u, v = np.mgrid[-1:1.5:0.1, -1:1.5:0.1]
	z = np.zeros(u.shape)

	for line, x1 in enumerate(u):
		for col, x2 in enumerate(v):
			y = 0
			cont = 0
			for i in range(0,degree):
				for j in range(0,i+1):
					y += (u[line][col]**(i-j))*(v[line][col]**(j))*theta[cont]
					cont += 1
			z[line][col] = sigmoid(y)

	return [u, v, z]


# reading data
file = open('ex2data2.txt');
data_in = []
X = []
y = []
for line in file:
	l = line.split(',')
	xi = [1]
	di = []
	for i in range(0, len(l)-1):
		xi.append(float(l[i]))
		di.append(float(l[i]))
	data_in.append(di)
	X.append(xi)
	y.append(int(l[len(l)-1].replace('\n', '')))

# mapping the initial values to a higher dimensional array
degree = 7
dim = int((degree*(degree+1))/2)
mapX = []
for i in range(len(X)):
	aux = []
	for j in range(dim):
		aux.append(0)
	mapX.append(aux)

cont = 0
for i in range(0,degree):
	for j in range(0,i+1):
		for k in range(len(X)):
			mapX[k][cont] = (X[k][1]**(i-j))*(X[k][2]**(j))
		cont += 1

# initializing parameters
theta = []
for i in range(0,len(mapX[0])):
	theta.append(0)

alpha = 1
costs = []
out = []

print("Initial cost:",cost(theta,mapX,y,1))

xNp = np.array(mapX)
yNp = np.array(y)

for i, lamb in enumerate([0, 1, 1000]):
	result = op.minimize(fun=cost,x0=theta,args=(xNp,yNp,lamb),method='TNC',jac=gradient)
	print("Final cost with lambda = ",lamb,":",cost(result.x,mapX,y,lamb))
	contr = generate_contour(result.x)
	plot_data(data_in,y,contr,i,lamb)

plt.show()