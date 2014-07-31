'''
Example of how to use Decision Trees and PCA in simpleml.
'''
import numpy as np

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml
# %cd M:/Libraries/Documents/Code/Python/simpleml

from simpleml.dectree import DecisionTree
from simpleml.ensemble import EnsembleBinaryClassifier

# Load data
x_data = np.loadtxt('x_data.txt')
y_data = np.loadtxt('y_class.txt', dtype=int)

num = 5

# Using untransformed data
x_train_list = [x_data[i*1000:(i+1)*1000] for i in range(num)]
y_train_list = [y_data[i*1000:(i+1)*1000] for i in range(num)]
x_test  = x_data[-1000:]
y_test  = y_data[-1000:]

trees = [DecisionTree(min_obs_split=5, max_depth=5) for i in range(num)]
for tree, x, y in zip(trees, x_train_list, y_train_list):
    tree.fit(x, y)

ensemble = EnsembleBinaryClassifier()
for tree in trees:
    ensemble.add_model(tree, weight=1)

print(np.mean(ensemble.classify(x_test) == y_test))
for tree in trees:
    print(np.mean(tree.classify(x_test) == y_test))
