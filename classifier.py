from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pykernels.graph.shortestpath2 import ShortestPath
from config import ConfigSVM
import sys
import numpy as np
import timeit
import datetime

from os import listdir
from os.path import isfile, join

import pygraphviz
import networkx as nx

def convert_to_numpy_matrix(g, size):
    res = np.zeros((size,size))
    for s, d, w in g.edges(data=True):
        label = ""
        label = g.number_of_edges(s, d)
        res[int(s)][int(d)] = int(label)
    return np.asmatrix(res)

def read_file(path):
    g = nx.Graph(pygraphviz.AGraph(path))
    res1 = convert_to_numpy_matrix(g,ConfigSVM.matrix_size)
    return res1

if (len(sys.argv) == 3):
    if (sys.argv[1] == "col"):
        pathToDataset = './../dataset/cologrind'    
    elif (sys.argv[1] == "call"):
        pathToDataset = './../dataset/callgrind'
    else:
        print "usage: call/col"
        sys.exit(1)

    file = open(join(pathToDataset, "maxsize"),'r')
    ConfigSVM.matrix_size = int(file.read())

    test_files = [join(".",str(sys.argv[2]))]

    train_sort_path = pathToDataset + '/sorting/'
    train_nonSort_path = pathToDataset + '/non-sorting/'
    train_sort_files = [train_sort_path + f for f in listdir(train_sort_path) if isfile(join(train_sort_path, f))]
    train_nonSort_files = [train_nonSort_path + f for f in listdir(train_nonSort_path) if isfile(join(train_nonSort_path, f))]

else:
	print "usage: python classifier call/col pathToTestFile"
	sys.exit(1)

print 'Finished to load paths'

##############################################

train_data = []
for p in train_sort_files:
    train_data.append(read_file(p).ravel())
for p in train_nonSort_files:
    train_data.append(read_file(p).ravel())

train_data = np.array(train_data)
print "train data ", train_data.shape

test_data = []
for p in test_files:
    test_data.append(read_file(p).ravel())
test_data = np.array(test_data)

##############################################
#reshape dataset
##############################################

nsamples, nx, ny = train_data.shape
train_data = train_data.reshape((nsamples,nx*ny))

print len(train_data), train_data.shape
print 'Finished to load training data'

##############################################
# print ShortestPath().gram(train_data)
##############################################

nsamples, nx, ny = test_data.shape
test_data = test_data.reshape((nsamples,nx*ny))

print len(test_data), test_data.shape
print 'Finished to load testing data'

start = timeit.default_timer()

##############################################

y = np.concatenate((np.ones(len(train_sort_files)), np.zeros(len(train_nonSort_files))))
print "Y", len(y)
classifier = SVC(kernel=ShortestPath())
classifier.fit(train_data, y)

print 'Finished to feed data'

stop = timeit.default_timer()
training_time = stop - start
print 'Training Time: ', training_time
start = timeit.default_timer()

##############################################

predicted = classifier.predict(test_data)

print 'Predictions:', predicted
stop = timeit.default_timer()
print 'Training Time: ', training_time
print 'Testing Time: ', stop - start