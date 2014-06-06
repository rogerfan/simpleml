'''
Example of how to use the DecisionTree module.

Author: Roger Fan
'''
import os
import time

import numpy as np

# %cd C:/Users/g1rxf01/Documents/Data/other/New folder/simpleml
os.chdir("C:/Users/g1rxf01/Documents/Data/other/New folder/simpleml")

from simpleml import DecisionTree

# Load data
x_data = np.loadtxt('x_data.txt')
y_class = np.loadtxt('y_class.txt', dtype=int)


tree = DecisionTree(x_data[:5000], y_class[:5000],
                    test_data=x_data[-5000:], test_labels=y_class[-5000:])

train_err = []
test_err = []
depth = []
for i in range(15):
    start = time.clock()
    tree.grow_tree(min_obs_split=5, max_depth=i)
    depth.append(i)
    train_err.append(tree.train_err())
    test_err.append(tree.test_err())

    print(i, time.clock() - start)

