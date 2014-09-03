'''
Example of how to use Decision Trees and PCA in simpleml.
'''
import time
import pickle
import gzip

import numpy as np

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml

from simpleml.dectree import DecisionTree
from simpleml.ensemble import RandomForest

# Load data
with gzip.open('./data/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

data = {}
names = ['train', 'valid', 'test']
for name, dataset in zip(names, (train_set, valid_set, test_set)):
    cond = np.logical_or(dataset[1] == 0, dataset[1] == 1)
    data[name] = (dataset[0][cond], dataset[1][cond])

rforest = RandomForest(min_obs_split=10, max_depth=5, max_features=28,
                   n_models_fit=30, seed=2349634)
dtree = DecisionTree(min_obs_split=10, max_depth=10, seed=2349634)

start = time.clock()
rforest.fit(data['train'][0][:2000], data['train'][1][:2000])
print("Forest time: {:6.2f}".format(time.clock() - start))

start = time.clock()
dtree.fit(data['train'][0][:2000], data['train'][1][:2000])
print("Tree time: {:6.2f}".format(time.clock() - start))

print("Errors:")
print(rforest.oob_error)
print(rforest.test_err(data['test'][0], data['test'][1]))
print(dtree.test_err(data['test'][0], data['test'][1]))
