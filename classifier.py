import numpy as np
from sklearn.svm import SVC
from config import ConfigSVM
from pykernels.graph.shortestpath2 import ShortestPath

import timeit
import datetime

import sys
from os import listdir
from os.path import isfile, join

import pygraphviz
import networkx as nx

def convert_to_numpy_matrix(g, size):
    res = np.zeros((size,size))
    for s, d, w in g.edges(data=True):
        res[int(s)][int(d)] = int(g.number_of_edges(s, d))
    return np.asmatrix(res)

def read_file(path):
    g = nx.Graph(pygraphviz.AGraph(path))
    return convert_to_numpy_matrix(g,ConfigSVM.matrix_size)

if (len(sys.argv) == 3):
    if (sys.argv[1] == "colo"):
        pathToDataset = './../dataset/cologrind'
    elif (sys.argv[1] == "call"):
        pathToDataset = './../dataset/callgrind'
    else:
        print 'usage: call/colo'
        sys.exit(1)

    file = open(join(pathToDataset, "maxsize"),'r')
    ConfigSVM.matrix_size = int(file.read())

    test_files = [join(".",str(sys.argv[2]))]

    train_sort_path = pathToDataset + '/sorting/'
    train_nonSort_path = pathToDataset + '/non-sorting/'
    train_sort_files = [train_sort_path + f for f in listdir(train_sort_path) if isfile(join(train_sort_path, f))]
    train_nonSort_files = [train_nonSort_path + f for f in listdir(train_nonSort_path) if isfile(join(train_nonSort_path, f))]

else:
	print 'usage: python classifier call/colo pathToTestFile'
	sys.exit(1)

print 'Finished to load paths'

##############################################

train_data = []
for p in train_sort_files:
    train_data.append(read_file(p).ravel())
for p in train_nonSort_files:
    train_data.append(read_file(p).ravel())
train_data = np.array(train_data)

test_data = []
for p in test_files:
    test_data.append(read_file(p).ravel())
test_data = np.array(test_data)

##############################################
#reshape dataset
##############################################

nsamples, nx, ny = train_data.shape
train_data = train_data.reshape((nsamples,nx*ny))
print 'Finished to load training data'

nsamples, nx, ny = test_data.shape
test_data = test_data.reshape((nsamples,nx*ny))

print 'Finished to load testing data'

start = timeit.default_timer()

##############################################

y = np.concatenate((np.ones(len(train_sort_files)), np.zeros(len(train_nonSort_files))))
classifier = SVC(kernel=ShortestPath())
classifier.fit(train_data, y)
print 'Finished to feed data'

stop = timeit.default_timer()
training_time = stop - start
print 'Training Time: ", training_time'
start = timeit.default_timer()

##############################################

predicted = classifier.predict(test_data)

print 'Predictions: ', predicted
stop = timeit.default_timer()
print 'Training Time: ', training_time
print 'Testing Time: ', stop - start
