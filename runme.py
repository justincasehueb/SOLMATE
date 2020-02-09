import numpy
import sklearn as sk
from sklearn.neural_network import MLPRegressor

import joblib

# read the file
readFile = open("solarData/exampleData.csv", "r")

# remove columns 5 and 13, and convert remainders to floats
# also convert the last column to 1 or 0
rows = (line.split(',') for line in readFile)
x = list(rows)

x.pop(0)

# convert to floats
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = float(x[i][j])

x = numpy.array(x)

#compute standard deviations and means
means = numpy.mean(x, axis=0)
stddev = numpy.std(x, axis=0)


#normalize training data
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = (x[i][j] - means[j]) / stddev[j]

filename = 'finalNN.sav'
mlp = joblib.load(filename)

x = numpy.nan_to_num(x)

pred = mlp.predict(x)
print(numpy.average(pred))