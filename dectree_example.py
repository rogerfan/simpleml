'''
Example of how to use Decision Trees and PCA in simpleml.
'''
import time

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# %cd C:/Users/g1rxf01/Downloads/New folder/simpleml
# %cd M:/Libraries/Documents/Code/Python/simpleml

from simpleml.dectree import DecisionTree
from simpleml.transform import PCA

# Load data
x_data = np.loadtxt('x_data.txt')
y_class = np.loadtxt('y_class.txt', dtype=int)

# Using untransformed data
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

logit = sm.Logit(y_train, x_train)
logit_result = logit.fit()
pred = logit_result.predict(x_test)
logit_testerr = 1 - np.mean(np.round(pred) == y_test)


# Using PCA with all components
x_data_pca = PCA(x_data[:11000], 50).transform()
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


# Using PCA with 10 components
x_data_pca = PCA(x_data[:11000], 10).transform()
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

logit = sm.Logit(y_train, x_train)
logit_result = logit.fit()
pred = logit_result.predict(x_test)
logit_pca1_testerr = 1 - np.mean(np.round(pred) == y_test)


# Plotting
fig1 = plt.figure(figsize=(8, 11))
ax1 = fig1.add_subplot(3, 1, 1)
ax1.plot(res['depth'], res['train_err'], label='Train, Unpruned')
ax1.plot(res['depth'], res['train_err_prune'], label='Train, Pruned')
ax1.plot(res['depth'], res['test_err'], label='Test, Unpruned')
ax1.plot(res['depth'], res['test_err_prune'], label='Test, Pruned')
ax1.axhline(logit_testerr, color='darkgrey', label='Test, Logit')
ax1.set_title('Untransformed')

ax2 = fig1.add_subplot(3, 1, 2)
ax2.plot(res_pca['depth'], res_pca['train_err'], label='Train, Unpruned')
ax2.plot(res_pca['depth'], res_pca['train_err_prune'], label='Train, Pruned')
ax2.plot(res_pca['depth'], res_pca['test_err'], label='Test, Unpruned')
ax2.plot(res_pca['depth'], res_pca['test_err_prune'], label='Test, Pruned')
ax2.axhline(logit_testerr, color='darkgrey', label='Test, Logit')
ax2.legend(bbox_to_anchor=(1.05, .5), loc='center left', borderaxespad=0.)
ax2.set_title('All PC')

ax3 = fig1.add_subplot(3, 1, 3)
ax3.plot(res_pca1['depth'], res_pca1['train_err'], label='Train, Unpruned')
ax3.plot(res_pca1['depth'], res_pca1['train_err_prune'], label='Train, Pruned')
ax3.plot(res_pca1['depth'], res_pca1['test_err'], label='Test, Unpruned')
ax3.plot(res_pca1['depth'], res_pca1['test_err_prune'], label='Test, Pruned')
ax3.axhline(logit_pca1_testerr, color='darkgrey', label='Test, Logit')
ax3.set_title('10 PC')

fig2 = plt.figure(figsize=(8, 4))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(res['depth'], res['test_err_prune'], label='Untransformed')
ax1.plot(res_pca['depth'], res_pca['test_err_prune'], label='All PC')
ax1.plot(res_pca1['depth'], res_pca1['test_err_prune'], label='10 PC')
ax1.axhline(logit_testerr, color='darkgrey', label='Logit')
ax1.axhline(logit_pca1_testerr, color='darkgrey', linestyle='--', label='Logit, 10 PC')
ax1.legend(loc='upper right')
ax1.set_title('Test Error, Pruned Trees')

plt.show()
