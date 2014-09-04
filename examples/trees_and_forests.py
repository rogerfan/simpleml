'''
Example of how to use Decision Trees and Random Forests.
'''
import time
import pickle
import gzip

import numpy as np
from sklearn.linear_model import LogisticRegression

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples

from simpleml.dectree import DecisionTree
from simpleml.ensemble import RandomForest

# Load data
with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()

data = {}
names = ['train', 'valid', 'test']
for name, dataset in zip(names, (train_set, valid_set, test_set)):
    cond = np.logical_or(dataset[1] == 0, dataset[1] == 1)
    data[name] = (dataset[0][cond], dataset[1][cond])

rforest = RandomForest(min_obs_split=20, max_depth=10, max_features=28,
                   n_models_fit=30, seed=2349634)
dtree = DecisionTree(min_obs_split=20, max_depth=10, seed=2349634)

start = time.perf_counter()
rforest.fit(*data['train'], verbose=True)
print("Forest time: {:6.2f}".format(time.perf_counter() - start))

start = time.perf_counter()
dtree.fit(*data['train'])
print("Tree time:   {:6.2f}".format(time.perf_counter() - start))

start = time.perf_counter()
logit = LogisticRegression(random_state=2349634).fit(*data['train'])
print("Logit time:  {:6.2f}".format(time.perf_counter() - start))


print("\nErrors:")
print("Forest (oob):  {:.5f}".format(rforest.oob_error))
print("Forest (test): {:.5f}".format(rforest.test_err(*data['test'])))
print("Tree   (test): {:.5f}".format(dtree.test_err(*data['test'])))
print("Logit  (test): {:.5f}".format(
    np.mean(logit.predict(data['test'][0]) != data['test'][1])))
