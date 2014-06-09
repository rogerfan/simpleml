'''
Example of how to use the DecisionTree object in simpleml.
'''
import time

import numpy as np
import matplotlib.pyplot as plt

# %cd C:/Users/g1rxf01/Documents/Data/other/New folder/simpleml
# %cd M:/Libraries/Documents/Code/Python/simpleml

from simpleml import DecisionTree

# Load data
x_data = np.loadtxt('x_data.txt')
y_class = np.loadtxt('y_class.txt', dtype=int)


x_train = x_data[:5000]
y_train = y_class[:5000]
x_cv    = x_data[5000:6000]
y_cv    = y_class[5000:6000]
x_test  = x_data[6000:11000]
y_test  = y_class[6000:11000]


tree = DecisionTree(x_train, y_train,
                    test_data=x_cv, test_labels=y_cv)

res = {}
res['train_err'] = []
res['train_err_prune'] = []
res['test_err'] = []
res['test_err_prune'] = []
res['depth'] = []
for i in range(20):
    start = time.clock()
    res['depth'].append(i)

    tree.grow(min_obs_split=5, max_depth=i)
    res['train_err'].append(tree.train_err())
    res['test_err'].append(tree.test_err(x_test,
                                         y_test))

    tree.prune()
    res['train_err_prune'].append(tree.train_err())
    res['test_err_prune'].append(tree.test_err(x_test,
                                               y_test))

    print(i, time.clock() - start)





def pca(num_comp, X):
    from scipy.linalg import svd
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    U, S, V = svd(X_std, full_matrices=False)
    return np.dot(X, V[:num_comp].T)


x_data_pca = pca(50, x_data[:11000])
x_train = x_data_pca[:5000]
x_cv    = x_data_pca[5000:6000]
x_test  = x_data_pca[6000:11000]

tree_pca = DecisionTree(x_train, y_train,
                        test_data=x_cv, test_labels=y_cv)

res_pca = {}
res_pca['train_err'] = []
res_pca['train_err_prune'] = []
res_pca['test_err'] = []
res_pca['test_err_prune'] = []
res_pca['depth'] = []
for i in range(20):
    start = time.clock()
    res_pca['depth'].append(i)

    tree_pca.grow(min_obs_split=5, max_depth=i)
    res_pca['train_err'].append(tree_pca.train_err())
    res_pca['test_err'].append(tree_pca.test_err(x_test,
                                                 y_test))

    tree_pca.prune()
    res_pca['train_err_prune'].append(tree_pca.train_err())
    res_pca['test_err_prune'].append(tree_pca.test_err(x_test,
                                                       y_test))

    print(i, time.clock() - start)


x_data_pca = pca(10, x_data[:11000])
x_train = x_data_pca[:5000]
x_cv    = x_data_pca[5000:6000]
x_test  = x_data_pca[6000:11000]

tree_pca1 = DecisionTree(x_train, y_train,
                         test_data=x_cv, test_labels=y_cv)

res_pca1 = {}
res_pca1['train_err'] = []
res_pca1['train_err_prune'] = []
res_pca1['test_err'] = []
res_pca1['test_err_prune'] = []
res_pca1['depth'] = []
for i in range(20):
    start = time.clock()
    res_pca1['depth'].append(i)

    tree_pca1.grow(min_obs_split=5, max_depth=i)
    res_pca1['train_err'].append(tree_pca1.train_err())
    res_pca1['test_err'].append(tree_pca1.test_err(x_test,
                                                   y_test))

    tree_pca1.prune()
    res_pca1['train_err_prune'].append(tree_pca1.train_err())
    res_pca1['test_err_prune'].append(tree_pca1.test_err(x_test,
                                                         y_test))

    print(i, time.clock() - start)






plt.plot(res['depth'], res['train_err'], label='Train, Unpruned')
plt.plot(res['depth'], res['train_err_prune'], label='Train, Pruned')
plt.plot(res['depth'], res['test_err'], label='Test, Unpruned')
plt.plot(res['depth'], res['test_err_prune'], label='Test, Pruned')
plt.legend(loc='upper right')
plt.show()

plt.plot(res_pca['depth'], res_pca['train_err'], label='Train, Unpruned')
plt.plot(res_pca['depth'], res_pca['train_err_prune'], label='Train, Pruned')
plt.plot(res_pca['depth'], res_pca['test_err'], label='Test, Unpruned')
plt.plot(res_pca['depth'], res_pca['test_err_prune'], label='Test, Pruned')
plt.legend(loc='upper right')
plt.show()


plt.plot(res_pca1['depth'], res_pca1['train_err'], label='Train, Unpruned')
plt.plot(res_pca1['depth'], res_pca1['train_err_prune'], label='Train, Pruned')
plt.plot(res_pca1['depth'], res_pca1['test_err'], label='Test, Unpruned')
plt.plot(res_pca1['depth'], res_pca1['test_err_prune'], label='Test, Pruned')
plt.legend(loc='upper right')
plt.show()
