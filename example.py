'''
Example of how to use Decision Trees and PCA in simpleml.
'''
import time
import numpy as np
from sklearn.datasets import load_digits

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml

from simpleml.dectree import DecisionTree
from simpleml.ensemble import RandomForest

# Load data
digits = load_digits(n_class=2)


rforest = RandomForest(min_obs_split=10, max_depth=10, max_features=8,
                   n_models_fit=30, seed=2349634)
dtree = DecisionTree(min_obs_split=10, max_depth=10, seed=2349634)

start = time.clock()
rforest.fit(digits.data, digits.target)
print(time.clock() - start)

start = time.clock()
dtree.fit(digits.data, digits.target)
print(time.clock() - start)

print(rforest.oob_error)
print(rforest.test_err(digits.data, digits.target))
print(dtree.test_err(digits.data, digits.target))
