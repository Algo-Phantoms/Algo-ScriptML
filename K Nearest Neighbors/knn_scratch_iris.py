import csv
import random

# load irist dataset and randomly split it into test set and training set

def LoadData(filename, split, training_data=[] , testing_data=[]):
	with open(filename, 'rt') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            training_data.append(dataset[x])
	        else:	
	            testing_data.append(dataset[x])


# Euclidean distance calcualtion

import math
def EuclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


import operator 
def clusters(training_data, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(training_data)):
		dist = EuclideanDistance(testInstance, training_data[x], length)
		distances.append((training_data[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


import operator
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


# MEASURING ACCURACY 

def accuracy(testing_data, predictions):
	correct = 0
	for x in range(len(testing_data)):
		if testing_data[x][-1] in predictions[x]: 
			correct = correct + 1
			
	return (correct/float(len(testing_data))*100) 

def main():
	# prepare data
	training_data=[]
	testing_data=[]
	split = 0.75
	LoadData('K Nearest Neighbors\iris.data', split, training_data, testing_data)
	print ('Train set: ' + repr(len(training_data)))
	print ('Test set: ' + repr(len(testing_data)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testing_data)):
		neighbors = clusters(training_data, testing_data[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testing_data[x][-1]))
	accuracy_score = accuracy(testing_data, predictions)
	print('Accuracy: ' + repr(accuracy_score) + '%')
	
main()