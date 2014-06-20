'''
Example of how to simulate data using numpy.
'''
import time

import numpy as np
import matplotlib.pyplot as plt


def interactions_contr(level, x_data, scale=1.):
    '''
    Calculates the contribution of an entire interactions level to the
    dependent variable in an additive model. Uses random (normally distributed)
    coefficients.

    Parameters
    ----------
    level : int
        Interactions level to use. E.g. 2 means all pairwise interactions,
        including the square of each variable.
    x_data : ndarray
        2-D array containing the data. Columns are variables while rows
        are observations.
    scale : float [1.]
        Scale to use for the random coefficients. Keep in mind that there are
        many interactions at higher levels, so you probably want this to
        get much smaller as the interactions go up to control the overall
        variance.

    Returns
    -------
    inter_contr : ndarray
        Contribution to the dependent variable from this interaction level.
    '''
    from itertools import combinations_with_replacement
    from numpy.random import normal

    index_combinations = combinations_with_replacement(range(x_data.shape[1]),
                                                       level)
    inter_contr = 0.
    for indices in index_combinations:
        inter = np.product(x_data[:, indices], axis=1)
        inter_contr += inter * normal(scale=scale)

    return inter_contr


# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml
# %cd M:/Libraries/Documents/Code/Python/simpleml

obs_num = 50000
var_num = 50
np.random.seed(452345)
start = time.clock()
print("Simulating {} observations with {} independent variables".format(
    obs_num, var_num
))

# Create a random covariance matrix for simulation
cov = np.cov(
    np.random.multivariate_normal(np.zeros(var_num), np.identity(var_num),
                                  size=int(var_num*1.5)),
    rowvar=0
)

# Simulate X data
x_data = np.random.multivariate_normal(np.zeros(var_num), cov, size=obs_num)

# Create initial Y data using random coefficients
y_data = np.dot(x_data, np.random.uniform(-1, 1, size=var_num))

# Add interaction effects for a couple levels of interactions
y_data += interactions_contr(2, x_data, scale=0.1)
y_data += interactions_contr(3, x_data, scale=0.01)

# Add random noise equal to about 9% (0.3**2) of the current variance
# y_data += np.random.normal(scale=0.3*y_data.std(), size=obs_num)

# Standardize data
x_data = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
y_data = (y_data - np.mean(y_data)) / np.std(y_data)

# Generate labels for classification
# y_class = np.floor(y_data) % 2 == 0
y_class = y_data >= 0

print('Simulation took {:.2f} seconds.'.format(time.clock() - start))

# Some visualizations
plt.scatter(x_data[:,0], y_data, c=y_class)
plt.show()

for i in range(0, var_num, 2):
    plt.scatter(x_data[:,i], x_data[:,i+1], c=y_class)
    plt.show()

# Save data
np.savetxt('x_data.txt', x_data, fmt='%.5e')
np.savetxt('y_data.txt', y_data, fmt='%.5e')
np.savetxt('y_class.txt', y_class, fmt='%i')
