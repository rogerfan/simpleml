'''
Decision Trees.
'''
import numpy as np

from . import metrics


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


class _DecisionNode:
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

    def classify_obs(self, obs):
        # Terminal condition
        if self.split is None:
            return self.majority

        # Recursively go down the tree
        var, split = self.split
        if obs[var] < split:
            return self.children[0].classify_obs(obs)
        else:
            return self.children[1].classify_obs(obs)

    def stumps(self):
        if self.children is None:
            return []
        elif self.children[0].split is None and self.children[1].split is None:
            return [self]
        else:
            return self.children[0].stumps() + self.children[1].stumps()

    def descendents(self):
        desc_list = [self]
        if self.split is not None:
            for i in range(2):
                desc_list += self.children[i].descendents()
        return desc_list


def _create_decision_tree(data, labels, min_obs_split=1, max_depth=None,
                          objfunc=metrics.gini):

    if max_depth is None:
        max_depth = np.inf

    prop = np.mean(labels)
    majority = int(np.round(prop))

    # Terminal conditions
    if (  len(labels) < min_obs_split or  # Leaf size condition
          max_depth == 0 or               # Reached max depth
          prop == 0 or prop == 1 ):       # Homogenous branch
        return _DecisionNode(majority=majority)

    # Find best split
    split = _choose_split(data, labels, objfunc)
    cond = data[:, split[0]] < split[1]

    # Build recursively defined tree
    tree = _DecisionNode(
        majority, split=split,
        children=[
            _create_decision_tree(
                data[cond], labels[cond],
                min_obs_split, max_depth-1, objfunc=objfunc
            ),
            _create_decision_tree(
                data[np.logical_not(cond)], labels[np.logical_not(cond)],
                min_obs_split, max_depth-1, objfunc=objfunc
            )
        ]
    )
    for child in tree.children:
        child.parent = tree
    return tree


def data_at_node(curr_node, target_node, data):

    if curr_node is target_node:
        return data
    if curr_node.split is None:
        return None

    var, split = curr_node.split
    result = data_at_node(curr_node.children[0], target_node,
                          data[data[:,var] < split])
    if result is None:
        result = data_at_node(curr_node.children[1], target_node,
                              data[data[:,var] >= split])

    return result


class DecisionTree:
    '''
    Decision Tree.

    Parameters
    ----------
    train_data, train_labels : ndarray [None, None]
        Independent data and labels for training.
    test_data, test_labels : ndarray [None, None]
        Independent data and labels for testing.
    test : _DecisionNode [None]
        Already grown tree.

    Attributes
    ----------
    data : dict
        Contains training and test data if provided.
    tree : _DecisionNode
        Contains the tree once it is grown
    grow_params : dict
        Contains parameters used for growing tree.
    '''
    def __init__(self, train_data=None, train_labels=None,
                 test_data=None, test_labels=None,
                 tree=None):

        self.data = {}
        self.tree = None
        self.pruned = False
        self.grow_params = None

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

    def copy(self):
        from copy import deepcopy

        args = self.data.copy()
        args['tree'] = deepcopy(self.tree)

        return DecisionTree(**args)

    def grow(self, min_obs_split=1, max_depth=None, objfunc=metrics.gini):
        '''
        Grow the decision tree using training data.

        Parameters
        ----------
        min_obs_split : int [1]
            Nodes with sizes less than this will not be split further.
        max_depth : int [None]
            Maximum depth to grow the tree to.
        objfunc : function [metrics.gini]
            Objective function to minimize when selecting splits.
        '''
        if max_depth is None:
            max_depth = np.inf

        # Grow tree
        self.tree = _create_decision_tree(
            self.data['train_data'], self.data['train_labels'],
            min_obs_split=min_obs_split, max_depth=max_depth, objfunc=objfunc
        )

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
            return self.tree.classify_obs(new_data)
        else:
            return np.array([self.tree.classify_obs(obs) for obs in new_data])

    def train_err(self):
        ''' Compute training error. '''
        if 'train_data' not in self.data:
            raise AttributeError('No training data provided.')

        return np.mean(self.classify(self.data['train_data']) !=
                       self.data['train_labels'])

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

        if 'test_data' not in self.data:
            raise AttributeError('No testing data provided.')

        return np.mean(self.classify(self.data['test_data']) !=
                       self.data['test_labels'])

    def prune(self):
        if self.tree is None:
            raise AttributeError('Tree has not been grown yet.')

        already_considered = []
        curr_err = self.test_err()

        pruned_something = True
        while pruned_something:
            pruned_something = False

            for stump in self.tree.stumps():
                if stump in already_considered:
                    continue

                # Set split to none, save split in case we need to replace it
                split = stump.split
                stump.split = None

                # Calculate new error after pruning the stump
                new_err = self.test_err()

                if new_err <= curr_err:
                    # Prune the stump
                    stump.children = None
                    curr_err = new_err
                    pruned_something = True
                else:
                    # Replace the stump
                    stump.split = split
                    already_considered.append(stump)

        return self

