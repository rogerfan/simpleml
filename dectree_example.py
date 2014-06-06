'''
Example of how to use the DecisionTree object in simpleml.
'''
import os
import time

import numpy as np
import matplotlib.pyplot as plt

# %cd C:/Users/g1rxf01/Documents/Data/other/New folder/simpleml
os.chdir("C:/Users/g1rxf01/Documents/Data/other/New folder/simpleml")

from simpleml import DecisionTree

# Load data
x_data = np.loadtxt('x_data.txt')
y_class = np.loadtxt('y_class.txt', dtype=int)


tree = DecisionTree(x_data[:5000], y_class[:5000],
                    test_data=x_data[-5000:], test_labels=y_class[-5000:])

train_err = []
train_err_prune = []
test_err = []
test_err_prune = []
depth = []
for i in range(20):
    start = time.clock()
    depth.append(i)

    tree.grow(min_obs_split=5, max_depth=i)
    train_err.append(tree.train_err())
    test_err.append(tree.test_err(x_data[-10000:-5000], y_class[-10000:-500]))

    tree.prune()
    train_err_prune.append(tree.train_err())
    test_err_prune.append(tree.test_err(x_data[-10000:-500], y_class[-10000:-500]))

    print(i, time.clock() - start)

plt.plot(depth, train_err, label='Train, Unpruned')
plt.plot(depth, train_err_prune, label='Train, Pruned')
plt.plot(depth, test_err, label='Test, Unpruned')
plt.plot(depth, test_err_prune, label='Test, Pruned')
plt.legend(loc='lower left')
plt.show()
