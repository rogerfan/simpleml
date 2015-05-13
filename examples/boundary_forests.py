'''
Example of how to use Boundary Forests.
'''
import os, sys
import time

import numpy as np
import matplotlib.pyplot as plt

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml/examples
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from simpleml.boundfor import BoundaryForest


# Sample random data
np.random.seed(2345)
num_obs = 500

x = np.random.normal(size=(num_obs, 2))
e = np.random.normal(1, size=num_obs)


### Choose a decision rule
# # Linear boundary with kink
# y = np.logical_and(np.dot(x, [ 2,-3])+e < 4,
#                    np.dot(x, [-3,-1])+e < 3).astype(int)

# Sinusoidal boundary
y = (x[:,1]+e*.8 < .8+np.sin((x[:,0])*3)).astype(int)

# # XOR
# x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([0, 1, 1, 0])


# Setup mesh for graphing boundary
x0_min, x1_min = np.min(x, axis=0)-.5
x0_max, x1_max = np.max(x, axis=0)+.5

x0, x1 = np.meshgrid(np.linspace(x0_min, x0_max, 100),
                     np.linspace(x1_min, x1_max, 100))
x_flatmesh = np.column_stack([x0.ravel(), x1.ravel()])

# Setup estimator
bf = BoundaryForest(maxchild=3, maxtrees=10)

# Estimate and plot
start = time.perf_counter()
bf.fit(x, y)
print('Estimation time: {:6.3f}'.format(time.perf_counter() - start))

start = time.perf_counter()
Z = bf.predict(x_flatmesh)
# Z = np.rint(Z)
print('Prediction time: {:6.3f}'.format(time.perf_counter() - start))

Z = Z.reshape(x0.shape)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.contour(x0, x1, Z, cmap=plt.cm.cool)
ax.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.cool)

fig.savefig('ex_bf_2d.pdf')
plt.close(fig)
