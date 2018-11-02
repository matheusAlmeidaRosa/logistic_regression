import numpy as np
import matplotlib.pyplot as plt
import random
import sys

r = 0
g = 1
b = 2

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
			dist = (centroid[r]-x[i][r])**2+(centroid[g]-x[i][g])**2+(centroid[b]-x[i][b])**2
			if (current_dist > dist):
				closest_centroid = idx
				current_dist = dist
		elements_of_each_centroid[closest_centroid].append(x[i])
	return elements_of_each_centroid

def computeMeans(elements_of_each_centroid, k):
	centroids = {}
	for i in range(k):
		new_centroid_r = 0
		new_centroid_g = 0
		new_centroid_b = 0
		for pixel in elements_of_each_centroid[i]:
			new_centroid_r += pixel[r]
			new_centroid_g += pixel[g]
			new_centroid_b += pixel[b]
		centroids[i] = (new_centroid_r/len(elements_of_each_centroid[i]), new_centroid_g/len(elements_of_each_centroid[i]), new_centroid_b/len(elements_of_each_centroid[i]))
	return centroids

def initializeDictElementsCentroid(centroids):
	elements_of_each_centroid = {}
	for i in range(k):
		elements_of_each_centroid[i] = []
	return elements_of_each_centroid

def toRGBFormat(image_matrix):
	rgb_format = []
	for i in range(image_matrix.shape[0]):
		for j in range(image_matrix.shape[1]):
				rgb_format.append((image_matrix[i][j][r],image_matrix[i][j][g],image_matrix[i][j][b],i,j))
	return rgb_format

def reconstructImageMatrix(elements_of_each_centroid, centroids, k, image_matrix):
	pos_x = 3
	pos_y = 4
	for i in range(k):
		colors = elements_of_each_centroid[i]
		for color in colors:
			image_matrix[color[pos_x]][color[pos_y]][r] = centroids[i][r]
			image_matrix[color[pos_x]][color[pos_y]][g] = centroids[i][g]
			image_matrix[color[pos_x]][color[pos_y]][b] = centroids[i][b]
	return image_matrix

image_matrix = plt.imread('bird_small.png')
pixels = toRGBFormat(image_matrix)

iterations = 10

k = 16
centroids = initCentroids(pixels, k)
for i in range(iterations):
	elements_of_each_centroid = initializeDictElementsCentroid(k)
	elements_of_each_centroid = findClosestCentroid(pixels, centroids, elements_of_each_centroid)
	centroids = computeMeans(elements_of_each_centroid, k)
	
imagem_matrix = reconstructImageMatrix(elements_of_each_centroid, centroids, k, image_matrix)
plt.imshow(image_matrix)
plt.show()