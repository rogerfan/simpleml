'''
Example of how to use Decision Trees and Random Forests.
'''
import sys, os

import numpy as np
import matplotlib.pyplot as plt

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.perceptron import MultilayerPerceptron


np.random.seed(2345)
num_obs = 500

x = np.random.normal(size=(num_obs, 2))
e = np.random.normal(1, size=num_obs)
y = np.logical_and(np.dot(x, [ 2,-3])+e < 4,
                   np.dot(x, [-3,-1])+e < 3).astype(int)




# w_inp, w_out = fit_nn(x, y, num_hid=4, epochnum=500, learn_rate=.5,
#                       momentum=0.1, seed=23456, verbose=True)
# y_pred = pred_nn(x, w_inp, w_out)

# x0_min, x1_min = np.min(x, axis=0)
# x0_max, x1_max = np.max(x, axis=0)

# x0, x1 = np.meshgrid(np.linspace(x0_min, x0_max, 100),
#                      np.linspace(x1_min, x1_max, 100))
# Z = pred_nn(np.column_stack([x0.ravel(), x1.ravel()]), w_inp, w_out)
# Z = Z.reshape(x0.shape)

# plt.contour(x0, x1, Z, cmap=plt.cm.Paired)
# plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.Paired)
