import numpy as np
import matplotlib.pyplot as plt
import random
import sys

def initCentroids(x, k):
	centroids = {}

	for i in range(k):
		newIdx = random.randint(0, len(x))
		while newIdx in centroids:
			newIdx = random.randint(0, len(x))
		centroids[i] = x[newIdx]
	return centroids

def findClosestCentroid(x, centroids, elements_of_each_centroid):
	for i in range(len(x)):
		closest_centroid = 0
		current_dist = sys.maxint
		for idx, centroid in centroids.items():
			dist = (centroid[0]-x[i][0])**2+(centroid[1]-x[i][1])**2
			if (current_dist > dist):
				closest_centroid = idx
				current_dist = dist
		elements_of_each_centroid[closest_centroid].append(x[i])
	return elements_of_each_centroid

def computeMeans(elements_of_each_centroid, k):
	centroids = {}
	for i in range(k):
		new_centroid_x = 0
		new_centroid_y = 0
		for point in elements_of_each_centroid[i]:
			new_centroid_x += point[0]
			new_centroid_y += point[1]
		centroids[i] = (new_centroid_x/len(elements_of_each_centroid[i]), new_centroid_y/len(elements_of_each_centroid[i]))
	return centroids

def initializeDictElementsCentroid(centroids):
	elements_of_each_centroid = {}
	for i in range(k):
		elements_of_each_centroid[i] = []
	return elements_of_each_centroid

file = open('data2.txt')
X = []
for line in file:
	l = line.split(',')
	t = (float(l[0]), float(l[1]))
	X.append(t)


plt.figure(1)
plt.title("k-means")
for point in X:
	plt.scatter(point[0], point[1], marker='.', color='r')


iterations = 10

k = 3
centroids = initCentroids(X, k)
for i in range(iterations):
	elements_of_each_centroid = initializeDictElementsCentroid(k)
	elements_of_each_centroid = findClosestCentroid(X, centroids, elements_of_each_centroid)
	centroids = computeMeans(elements_of_each_centroid, k)

for key, point in centroids.items():
	plt.scatter(point[0], point[1], marker='x', color='b')

plt.show()