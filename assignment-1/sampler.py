#! /usr/bin/python
#
# Machine Learning - Assignment 1
# Astegher Maurizio 175195

import random

dataset = {} # dataset[0] -> setosa, dataset[1] -> versicolor, dataset[2] -> virginica
train = {}
test = {}
N = 3 # Number of training and test sets to be generated

# Read "iris.dat" file
with open("iris.dat", "r") as f:
	lines = f.read().splitlines()

# Split data according to the "type" attribute
for i in range(0, 3):
	dataset[i] = []
for i in range(1, 51):
	dataset[0].append(lines[i])
for i in range(51, 101):
	dataset[1].append(lines[i])
for i in range(101, 151):
	dataset[2].append(lines[i])

# Generate N training sets and N test sets
for i in range(0, N):
	train[i] = []
	test[i] = []

	# Randomly sample a training set (33 examples of each type) and a test set (the remaining)
	for j in dataset:
		temp = range(0, 50)
		for k in range(0, 33): 
			r = random.choice(temp)
			temp.remove(r)
			train[i].append(dataset[j][r])
		for k in temp:
			test[i].append(dataset[j][k])

# Randomly move a test example to the training set (one is in excess)
for i in range(0, N):
	r = random.choice(test[i])
	test[i].remove(r)
	train[i].append(r)

# Write training and test sets to file
for i in range(0, N):
	trainFile = open("train_" + str(i+1) + ".dat", "w")
	testFile = open("test_" + str(i+1) + ".dat", "w")
	
	trainFile.write(lines[0] + "\n")
	testFile.write(lines[0] + "\n")
	for j in train[i]:
		trainFile.write(j + "\n")
	for j in test[i]:
		testFile.write(j + "\n")

	trainFile.close()
	testFile.close()