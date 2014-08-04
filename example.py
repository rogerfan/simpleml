'''
Example of how to use Decision Trees and PCA in simpleml.
'''
import time
import numpy as np

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml
# %cd M:/Libraries/Documents/Code/Python/simpleml

from simpleml.dectree import DecisionTree
from simpleml.ensemble import BaggingBinaryClassifier

# Load data
x_data = np.loadtxt('x_data.txt')
y_data = np.loadtxt('y_class.txt', dtype=int)

x_train = x_data[:5000]
y_train = y_data[:5000]
x_test  = x_data[-5000:]
y_test  = y_data[-5000:]

model_params = {
    'min_obs_split': 10,
    'max_depth': 10,
    'max_features': 7
}

bag = BaggingBinaryClassifier(DecisionTree, model_params=model_params,
                              n_models_fit=40, seed=2349634)
dtree = DecisionTree(min_obs_split=10, max_depth=10)

start = time.clock()
bag.fit(x_train, y_train)
print(time.clock() - start)

start = time.clock()
dtree.fit(x_train, y_train)
dtree.prune(x_test, y_test)
print(time.clock() - start)


print(bag.test_err(x_test, y_test))
print(dtree.test_err(x_test, y_test))
