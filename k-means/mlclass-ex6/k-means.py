import numpy as np
import matplotlib.pyplot as plt
import random
import sys

def initCentroids(x, k):
	centroids = []

	for i in range(k):
		newIdx = random.randint(0, len(x[0]))
		while newIdx in centroids:
			newIdx = random.randint(0, len(x[0]))
		centroids.append(newIdx)
	return centroids

def findClosestCentroid(x, centroids):
	xCent = []
	for i in len(x[0]):
		small = 0
		current_dist = sys.maxint
		for cent in centroids:
			dist = (x[0][cent]-x[0][i])**2+(x[1][cent]-x[1][i])**2
			if (current_dist > dist):
				small = cent
				current_dist = dist
		xCent.append(small)
	return xCent

def computeMeans(x, idx, k):
	pass

file = open('data2.txt');
data_in = []
x = []
y = []
for line in file:
	l = line.split(',')
	x.append(float(l[0]))
	y.append(float(l[1]))

X = [x, y]

"""plt.figure(1)
plt.title("k-means")
plt.scatter(x, y, marker='.')
plt.show()"""

iterations = 0

k = 4
centroids = initCentroids(X, k)
print(centroids)
for i in range(iterations):
	idx = findClosestCentroid(X, centroids)
	centroids = computeMeans(X, idx, K)
