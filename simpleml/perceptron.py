'''
Multi-layer perceptrons.
'''
from copy import deepcopy

import numpy as np

from . import metrics
from .helpers import np_print_options

__all__ = ('MultilayerPerceptron',)


class _Layer:
    def __init__(self, num_inputs, num_nodes, parent=None,
                 sigmoid=metrics.logistic, weight_init_range=.5):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.sigmoid = sigmoid

        self.parent = parent
        self.child = None
        if parent is not None:
            parent.child = self

        self._activation_scores = np.zeros(num_nodes)
        self.activations = np.zeros(num_nodes)
        self.weights = np.random.uniform(-weight_init_range, weight_init_range,
                                         size=(num_inputs, num_nodes))
        self._last_change = np.zeros((num_inputs, num_nodes))
        self._deltas = np.zeros(num_nodes)

    def __str__(self):
        with np_print_options(precision=5, suppress=True):
            result = "  {}\n{}".format(
                str(self.weights).replace('\n', '\n  '),
                self.activations
            )
        return result

    def update_activations(self, inputs):
        self._activation_scores = np.dot(inputs, self.weights)
        self.activations = self.sigmoid.f(self._activation_scores)

        if self.child is None:
            return self.activations
        else:
            return self.child.update_activations(self.activations)

    def backpropogate(self, system_inputs, errors, learn_rate, momentum):
        if self.parent is None:
            inputs = system_inputs
        else:
            inputs = self.parent.activations

        self._deltas = self.sigmoid.d(self._activation_scores)*errors
        self._last_change = (np.outer(inputs, self._deltas) +
                             momentum*self._last_change)
        self.weights += learn_rate*self._last_change

        if self.parent is None:
            return np.dot(self._deltas, self.weights.T)
        else:
            return self.parent.backpropogate(
                system_inputs, np.dot(self._deltas, self.weights.T),
                learn_rate, momentum
            )


class MultilayerPerceptron:
    '''
    Multilayer Perceptron.

    Parameters
    ----------
    num_inputs : int, optional
        Number of input nodes to the system. Should be the same as the number
        of features of input data, +1 if you use add_constant=True in the
        fit method (default 2).
    num_outputs : int, optional
        Number of output nodes from the system. Should be the same as the number
        of variables in the training output data (default 1).
    num_hidden_layers : int, optional
        Number of hidden layers to train (default 1).
    num_hidden_nodes : int or list, optional
        Number of nodes per hidden layer. If an int is provided then all the
        hidden layers will have that number of hidden nodes. If a list-like
        is provided then they will be mapped onto the corresponding layers
        ordered from input to output (default 3).
    learn_rate : float, optional
        Learning rate for training (default 0.5).
    momentum : float, optional
        Momentum parameter for training (default 0.1).
    seed : int, optional
        Seeds the random number generator (default None).
    sigmoid : object, optional
        Sigmoid function to use. Must have f and d methods for the function
        and derivative values, respectively (default metrics.logistic).
    weight_init_range : float, optional
        Weights are initalized from a uniform distribution between plus-minus
        this value (default .5).
    '''
    params_names = (
        'num_inputs', 'num_outputs', 'num_hidden_layers', 'num_hidden_nodes',
        'learn_rate', 'momentum', 'seed', 'sigmoid',
        'weight_init_range'
    )

    def __init__(self, num_inputs=2, num_outputs=1,
                 num_hidden_layers=1, num_hidden_nodes=3,
                 learn_rate=.5, momentum=.1, seed=None,
                 sigmoid=metrics.logistic, weight_init_range=.5):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learn_rate = .5
        self.momentum=.1
        self.seed = seed
        self.sigmoid = sigmoid
        self.weight_init_range = weight_init_range

        self.num_hidden_layers = num_hidden_layers
        if isinstance(num_hidden_nodes, int):
            num_hidden_nodes = [num_hidden_nodes
                                for i in range(num_hidden_layers)]
        elif len(num_hidden_nodes) != num_hidden_layers:
            raise ValueError(
                'Length of num_hidden_nodes ({}) must match num_hidden_layers '
                '({}).'.format(len(num_hidden_nodes), num_hidden_layers)
            )
        self.num_hidden_nodes = num_hidden_nodes

        lpars = {'sigmoid': sigmoid, 'weight_init_range': weight_init_range}

        self.layers = []
        self.layers.append(_Layer(num_inputs, num_hidden_nodes[0], **lpars))
        for num_inp, num_nodes in zip(num_hidden_nodes[:-1],
                                      num_hidden_nodes[1:]):
            self.layers.append(_Layer(num_inp, num_nodes,
                                      parent=self.layers[-1], **lpars))
        self.layers.append(_Layer(num_hidden_nodes[-1], num_outputs,
                                  parent=self.layers[-1], **lpars))


    def __str__(self):
        result = '[ '+  ' '.join(['x' for _ in range(self.num_inputs)]) + ' ]'
        for l in self.layers:
            result += '\n' + str(l)
        return result

    @property
    def params(self):
        result = {}
        for name in self.params_names:
            result[name] = getattr(self, name)
        return result

    def copy(self):
        result = MultilayerPerceptron(**self.params)
        result.layers = deepcopy(self.layers)
        return result

    def fit(self, X, Y, epochnum=1000):
        '''
        Fit the multilayer perceptron using training data.

        Parameters
        ----------
        epochnum : int, optional
            Number of epochs, i.e. passes through the entire dataset (default 1000).
        '''
        pass
