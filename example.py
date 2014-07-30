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
y_class = np.loadtxt('y_class.txt', dtype=int)

# Using untransformed data
x_train1 = x_data[:1000]
y_train1 = y_class[:1000]
x_train2 = x_data[1000:2000]
y_train2 = y_class[1000:2000]
x_train3 = x_data[2000:3000]
y_train3 = y_class[2000:3000]
x_train4 = x_data[3000:4000]
y_train4 = y_class[3000:4000]
x_test  = x_data[-1000:]
y_test  = y_class[-1000:]

tree1 = DecisionTree(x_train1, y_train1)
tree2 = DecisionTree(x_train2, y_train2)
tree3 = DecisionTree(x_train3, y_train3)
tree4 = DecisionTree(x_train4, y_train4)
tree1.fit(min_obs_split=5, max_depth=5)
tree2.fit(min_obs_split=5, max_depth=5)
tree3.fit(min_obs_split=5, max_depth=5)
tree4.fit(min_obs_split=5, max_depth=5)

ensemble = EnsembleBinaryClassifier()
ensemble.add_model(tree1, weight=1)
ensemble.add_model(tree2, weight=2)
ensemble.add_model(tree3, weight=2)
ensemble.add_model(tree4, weight=2)

print(np.mean(ensemble.classify(x_test) == y_test))
print(np.mean(tree1.classify(x_test) == y_test))
print(np.mean(tree2.classify(x_test) == y_test))
print(np.mean(tree3.classify(x_test) == y_test))
print(np.mean(tree4.classify(x_test) == y_test))
