'''
Example of how to use Decision Trees and Random Forests.
'''
import os, sys
import time
import pickle
import gzip

import numpy as np
from sklearn.linear_model import LogisticRegression

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.dectree import DecisionTree
from simpleml.ensemble import RandomForest

# Load data
with gzip.open('../data/mnist.gz', 'rb') as f:
    raw_data = pickle.load(f)

data = {}
for key, val in raw_data.items():
    cond = np.logical_or(val[1] == 0, val[1] == 1)
    data[key] = (val[0][cond], val[1][cond])


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
