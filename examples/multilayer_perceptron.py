'''
Example of how to use Decision Trees and Random Forests.
'''
import sys, os

import numpy as np
import matplotlib.pyplot as plt

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.perceptron import Layer
from simpleml.baseclasses import Function


np.random.seed(2345)
num_obs = 500

x = np.random.normal(size=(num_obs, 2))
e = np.random.normal(1, size=num_obs)
y = np.logical_and(np.dot(x, [ 2,-3])+e < 4,
                   np.dot(x, [-3,-1])+e < 3).astype(int)


