'''
Implementation of Decision Trees.

Author: Roger Fan
'''
import numpy as np

from . import objfunc


def _choose_split(data, labels, objfunc):
    min_obj = np.inf
    for cand_var in range(data.shape[1]):
        for cand_split in np.unique(data[:,cand_var])[1:]:
            subset0 = labels[data[:,cand_var] <  cand_split]
            subset1 = labels[data[:,cand_var] >= cand_split]
            frac0 = len(subset0)/len(labels)
            cand_obj = ( objfunc(subset0) * frac0 +
                         objfunc(subset1) * (1-frac0) )
            if cand_obj < min_obj:
                min_obj = cand_obj
                min_split = (cand_var, cand_split)

    return min_split


def create_decision_tree(data, labels, min_obs_split=1, max_depth=None,
                         objfunc=objfunc.gini):

    if max_depth is None:
        max_depth = np.inf

    prop = np.mean(labels)
    majority = int(np.round(prop))

    # Terminal conditions
    if (  len(labels) < min_obs_split or  # Leaf size condition
          max_depth == 0 or               # Reached max depth
          prop == 0 or prop == 1 ):       # Homogenous branch
        return DecisionNode(majority=majority)

    # Find best split
    split = _choose_split(data, labels, objfunc)
    cond = data[:, split[0]] < split[1]

    # Build recursively defined tree
    tree = DecisionNode(
        majority, split=split,
        children=[
            create_decision_tree(
                data[cond], labels[cond],
                min_obs_split, max_depth-1, objfunc=objfunc
            ),
            create_decision_tree(
                data[np.logical_not(cond)], labels[np.logical_not(cond)],
                min_obs_split, max_depth-1, objfunc=objfunc
            )
        ]
    )
    for child in tree.children:
        child.parent = tree
    return tree


def classify_obs(obs, tree):
    # Terminal condition
    if tree.children is None:
        return tree.majority

    # Recursively go down the tree
    var, split = tree.split
    if obs[var] < split:
        return classify_obs(obs, tree.children[0])
    else:
        return classify_obs(obs, tree.children[1])


class DecisionNode:
    def __init__(self, majority, split=None, children=None, parent=None):
        self.split = split
        self.majority = majority
        self.children = children
        self.parent = parent

    def __str__(self, level=0):
        if self.split is None:
            splitrep = 'Leaf'
        else:
            splitrep = self.split

        ret = '| '*level + '{}: {}\n'.format(splitrep, self.majority)
        if self.children is not None:
            for child in self.children:
                ret += child.__str__(level+1)
        return ret


class DecisionTree:
    '''
    Decision Tree implementation.

    Parameters
    ----------
    train_data, train_labels : ndarray [None, None]
        Independent data and labels for training.
    test_data, test_labels : ndarray [None, None]
        Independent data and labels for testing.
    test : DecisionNode [None]
        Already grown tree.

    Attributes
    ----------
    data : dict
        Contains training and test data if provided.
    tree : DecisionNode
        Contains the tree once it is grown
    grow_params : dict
        Contains parameters used for growing tree.
    '''
    data = {}
    tree = None
    grow_params = None
    _err = {'train': None, 'test': None}

    def __init__(self, train_data=None, train_labels=None,
                 test_data=None, test_labels=None,
                 tree=None):

        def check_dim(data, labels):
            assert(len(data) == len(labels))
            assert(len(labels.shape) == 1)
            assert(len(data.shape) == 2)

        if train_data is not None and train_labels is not None:
            check_dim(train_data, train_labels)

            self.data['train_data'] = train_data
            self.data['train_labels'] = train_labels

        if test_data is not None and test_labels is not None:
            check_dim(test_data, test_labels)

            self.data['test_data'] = test_data
            self.data['test_labels'] = test_labels

        if tree is not None:
            self.tree = tree

    def __str__(self):
        return self.tree.__str__()

    def grow_tree(self, min_obs_split=1, max_depth=None, objfunc=objfunc.gini):
        '''
        Grow the decision tree using training data.

        Parameters
        ----------
        min_obs_split : int [1]
            Nodes with sizes less than this will not be split further.
        max_depth : int [None]
            Maximum depth to grow the tree to.
        objfunc : function [objfunc.gini]
            Objective function to minimize when selecting splits.
        '''
        if max_depth is None:
            max_depth = np.inf

        # Grow tree
        self.tree = create_decision_tree(
            self.data['train_data'], self.data['train_labels'],
            min_obs_split=min_obs_split, max_depth=max_depth, objfunc=objfunc
        )

        # Reset errors
        self._err['train'] = None
        if 'test' in self._err: self._err['test'] = None

        # Save parameters
        self.grow_params = {
            'min_obs_split': min_obs_split,
            'max_depth': max_depth,
            'objfunc': objfunc
        }

        return self

    def classify(self, new_data):
        '''
        Use the grown decision tree to classify data.

        Parameters
        ----------
        new_data : ndarray
            Array of new data. Can be a single or multiple observations.
        '''
        if self.tree is None:
            raise AttributeError('Tree has not been grown yet.')

        if len(new_data.shape) == 1:
            return classify_obs(new_data, self.tree)
        else:
            return np.array([classify_obs(obs, self.tree) for obs in new_data])

    def train_err(self):
        ''' Compute training error. '''
        self._err['train'] = self._calc_error(errtype='train')
        return self._err['train']

    def test_err(self, data=None, labels=None):
        '''
        Compute test error. Will use previously provided test data, or can
        be provided with new test data.

        Parameters
        ----------
        data, labels : ndarray [None, None]
            Data to calculate test error on. If not provided, will attempt to
            use test data provided at initialization.
        '''
        if data is not None and labels is not None:
            return np.mean(self.classify(data) != labels)

        self._err['test'] = self._calc_error(errtype='test')
        return self._err['test']

    def _calc_error(self, errtype='train'):
        if '{}_data'.format(errtype) not in self.data:
            raise AttributeError('No {}ing data provided.'.format(errtype))
        elif self._err[errtype] is not None:
            return self._err[errtype]

        pred = self.classify(self.data['{}_data'.format(errtype)])
        return np.mean(pred != self.data['{}_labels'.format(errtype)])



