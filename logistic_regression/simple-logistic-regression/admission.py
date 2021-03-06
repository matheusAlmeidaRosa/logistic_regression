"""
	This program realize a logistic regression for a dataset of students. 
	The dataset is composed by two past notes and the result (approved or not) of thes students.
	The logistic regression will try to divide the dataset in approved and reproved students.

	Author: Higor Coimbra - Matheus Rosa
	Class: Computational Inteligence
	University: CEFET-MG
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as op
import sys

import math

eps = 0.0000001

def plot_data(X, y, theta, line):
	plt.figure(1)
	plt.title("Simple Logistic Regression")
	plt.xlabel("Exam 1 score")
	plt.ylabel("Exam 2 score")

	n1 = False
	n2 = False
	for i, point in enumerate(X):
		if y[i]:
			marker = 'k+'
			label = 'Admitted'
		else:
			marker = 'yo'
			label = 'Not admitted'
		if y[i] and not n1:
			plt.plot(point[0], point[1], marker, label=label)
			n1 = True
		elif not y[i] and not n2:
			plt.plot(point[0], point[1], marker, label=label)
			n2 = True
		else:
			plt.plot(point[0], point[1], marker)

	plt.plot(line[0], line[1], color='r')
	plt.legend(loc=1)
	plt.show()

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

def cost(theta, X, y):
	m = len(X)
	summ = 0
	for i in range(0,m):
		h = output(theta,X[i])
		summ += (-y[i]*math.log(max(eps,h)) - (1.0-y[i])*math.log(max(eps, 1-h)))
	return (1/m)*summ

def gradient(theta, X, y):
	m = len(X)
	theta_new = []
	for j in range(0,len(theta)):
		summ = 0
		for i in range(0,m):
			summ += (output(theta,X[i])-y[i])*X[i][j]
		theta_new.append((1/m)*summ)
	return theta_new

def generate_line(theta, X):
	x1 = float('inf')
	x2 = -x1
	for i in range(0,len(X)):
		x1 = min(x1, X[i][1])
		x2 = max(x2, X[i][1])
	plot_x = [x1, x2]
	y1 = (-1/theta[2])*(theta[0]+theta[1]*x1)
	y2 = (-1/theta[2])*(theta[0]+theta[1]*x2)
	plot_y = [y1, y2]
	return [plot_x, plot_y]

# reading data
file = open('ex2data1.txt');
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

# initializing parameters
theta = []
for i in range(0,len(X[0])):
	theta.append(0)

iterations = 400
alpha = 1
costs = []
out = []

print("Initial cost:",cost(theta,X,y))

xNp = np.array(X)
yNp = np.array(y)

result = op.minimize(fun=cost,x0=theta,args=(xNp,yNp),method='TNC',jac=gradient)
print("Final cost:",cost(result.x,X,y))
print("Thetas:",result.x)

line = generate_line(result.x, X)
plot_data(data_in,y,theta,line)