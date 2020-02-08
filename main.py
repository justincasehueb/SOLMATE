import numpy
import sklearn as sk
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import csv
import random



###QUESTION 1###

# splits according to a given size of first array
def split(list, size):
    return list[:size], list[size:]

# read the file
readFile = open("frankenstein_data.csv", "r")

# remove columns 5 and 13, and convert remainders to floats
# also convert the last column to 1 or 0
rows = (line.split(',') for line in readFile)
x = list(rows)

x.pop(0)

# convert to floats
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = float(x[i][j])

#shuffle
#random.shuffle(x)

#split off 20% for test
test, x = split(x,10)
print(test)

#split remaining into validation and training groups
val, train = split(x, 10)

#confirmation of sizes
print("Test:", len(test), "Validation:", len(val), "Training:", len(train), "\n")

#create y values for train
trainY = []
for xVal in train:
    trainY.append(xVal[-1])
    xVal.pop(-1)

#create y values for validation
valY = []
for xVal in val:
    valY.append(xVal[-1])
    xVal.pop(-1)

#create y values for testing
testY = []
for xVal in test:
    testY.append(xVal[-1])
    xVal.pop(-1)

#convert all to numpy arrays
train = numpy.array(train)
trainY = numpy.array(trainY)

val = numpy.array(val)
valY = numpy.array(valY)

test = numpy.array(test)
testY = numpy.array(testY)

#concatenate training and validation for easy calculation
trainVal = numpy.concatenate((train, val), axis=0)

#compute standard deviations and means
means = numpy.mean(trainVal, axis=0)
stddev = numpy.std(trainVal, axis=0)

'''
#normalize training data
for i in range(len(train)):
    for j in range(len(train[i])):
        train[i][j] = (train[i][j] - means[j]) / stddev[j]

#normalize validation data
for i in range(len(val)):
    for j in range(len(val[i])):
        val[i][j] = (val[i][j] - means[j]) / stddev[j]

#normalize testing data
for i in range(len(test)):
    for j in range(len(test[i])):
        test[i][j] = (test[i][j] - means[j]) / stddev[j]
'''

##QUESTION 2##

#NEURAL NETS#

#created neural net
mlp = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes= 1000)

#fit on training data
mlp.fit(train, trainY)

#predict training data
trainPredNN = mlp.predict(train)

#predict validation data
valPredNN = mlp.predict(val)

#predict test data
testPredNN = mlp.predict(test)

testLen = numpy.linspace(0, len(testY) - 1, len(testY))

goodUp = 0
goodDown = 0
bad = 0
for i in range(len(testY)):
    if (testPredNN[i] > test[i,0] and testY[i] > test[i,0]):
        goodUp += 1
    elif (testPredNN[i] < test[i,0] and testY[i] < test[i,0]):
        goodDown += 1
    else:
        bad += 1

#print("Good high:", goodUp, "Good low:", goodDown, "Bad:", bad)
plt.plot(testY)
plt.scatter(testLen,testPredNN)
plt.show()

