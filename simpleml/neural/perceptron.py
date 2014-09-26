'''
Multi-layer perceptrons.
'''
from copy import deepcopy

import numpy as np

from .. import metrics
from ..helpers import np_print_options, check_random_state

__all__ = ('MultilayerPerceptron',)


def _constant(i):
    return 1
def _sqrt(i):
    return 1 / (i+1)**.5
def _linear(i):
    return 1/(i+1)
def _quadratic(i):
    return 1/((i+1)**2)


class _Layer:
    def __init__(self, num_inputs, num_nodes, parent=None, bias=True,
                 sigmoid=metrics.logistic, weight_init_range=.5,
                 seed=None):
        random = check_random_state(seed)

        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.sigmoid = sigmoid
        self.bias = bias

        self.parent = parent
        self.child = None
        if parent is not None:
            parent.child = self

        if bias:
            num_nodes_nobias = num_nodes - 1
        else:
            num_nodes_nobias = num_nodes

        self._activation_scores = np.zeros(num_nodes_nobias)
        self.activations = np.zeros(num_nodes)
        if bias: self.activations[0] = 1
        self.weights = random.uniform(-weight_init_range, weight_init_range,
                                      size=(num_inputs, num_nodes_nobias))
        self._last_change = np.zeros((num_inputs, num_nodes_nobias))
        self._deltas = np.zeros(num_nodes_nobias)

    def __str__(self):
        layer_rep = ('[ ' +
                     self.bias*'1 ' + (1-self.bias)*'x ' +
                     ' '.join(['x' for _ in range(self.num_nodes-1)]) +
                     (self.num_nodes > 1)*' ' +
                     ']')

        with np_print_options(precision=5, suppress=True):
            result = "  {}\n{}".format(
                str(self.weights).replace('\n', '\n  '),
                layer_rep
            )
        return result

    def update_activations(self, inputs):
        self._activation_scores = np.dot(inputs, self.weights)
        if self.bias:
            if len(inputs.shape) == 1:
                self.activations = np.append(
                    [1], self.sigmoid.f(self._activation_scores))
            else:
                self.activations = np.hstack(
                    [np.ones(len(inputs)).reshape(-1,1),
                     self.sigmoid.f(self._activation_scores)])
        else:
            self.activations = self.sigmoid.f(self._activation_scores)

        return self.activations

    def backpropogate(self, system_inputs, errors, learn_rate, momentum):
        if self.parent is None:
            inputs = system_inputs
        else:
            inputs = self.parent.activations

        if self.bias:
            self._deltas = self.sigmoid.d(self._activation_scores)*errors[1:]
        else:
            self._deltas = self.sigmoid.d(self._activation_scores)*errors

        change = np.outer(inputs, self._deltas)
        self._last_change = change + momentum*self._last_change
        self.weights += learn_rate*self._last_change

        return np.dot(self._deltas, self.weights.T)


class MultilayerPerceptron:
    '''
    Multilayer Perceptron.

    Parameters
    ----------
    num_inputs : int, optional
        Number of input nodes to the system. Should be the same as the number
        of features of input data, +1 if you use add_constant=True in the
        fit method (default 3).
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
    learn_rate_evol : function or {constant', 'sqrt', linear', 'quadratic'}
        Learning rate evolution function that is multiplied by learn_rate.
        Should be a single-input function that is defined for non-negative
        integers, returns values between 0 and 1, and is weakly decreasing
        (default 'linear').
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
        'learn_rate', 'learn_rate_evol', 'momentum', 'seed', 'sigmoid',
        'weight_init_range'
    )

    def __init__(self, num_inputs=3, num_outputs=1,
                 num_hidden_layers=1, num_hidden_nodes=3,
                 learn_rate=.5, learn_rate_evol='linear', momentum=.1,
                 seed=None, sigmoid=metrics.logistic, weight_init_range=.5):
        self.epochs_estimated = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.sigmoid = sigmoid
        self.weight_init_range = weight_init_range

        self.seed = seed
        self._random = check_random_state(seed)

        if callable(learn_rate_evol):
            self.learn_rate_evol = learn_rate_evol
        elif learn_rate_evol == 'constant':
            self.learn_rate_evol = _constant
        elif learn_rate_evol == 'sqrt':
            self.learn_rate_evol = _sqrt
        elif learn_rate_evol == 'linear':
            self.learn_rate_evol = _linear
        elif learn_rate_evol == 'quadratic':
            self.learn_rate_evol = _quadratic
        else:
            raise ValueError(
                '{} is not a recognized learn_rate_evol.'.format(
                    learn_rate_evol)
            )

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

        lpars = {'sigmoid': sigmoid, 'weight_init_range': weight_init_range,
                 'seed':self._random}

        self.layers = []
        self.layers.append(_Layer(num_inputs, num_hidden_nodes[0],
                                  bias=True, **lpars))
        for num_inp, num_nodes in zip(num_hidden_nodes[:-1],
                                      num_hidden_nodes[1:]):
            self.layers.append(
                _Layer(num_inp, num_nodes, parent=self.layers[-1],
                       bias=True, **lpars)
            )
        self.layers.append(_Layer(num_hidden_nodes[-1], num_outputs, bias=False,
                                  parent=self.layers[-1], **lpars))


    def __str__(self):
        result = ('[ 1 ' +
                  ' '.join(['x' for _ in range(self.num_inputs-1)]) +
                  ' ]')
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


    def fit(self, X, Y, epochnum=100, add_constant=True, verbose=False):
        '''
        Fit the multilayer perceptron using training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature dataset for training. Either a constant column is required
            or add_constant must be set to True. If there is already a constant
            column then it must be the first column in X.
        Y : array of shape (n_samples, n_outputs)
            Output data for training.
        epochnum : int, optional
            Number of epochs, i.e. passes through the entire dataset
            (default 1000).
        add_constant : bool, optional
            Adds a column of ones to the front of the X data. Set to False
            if your data already has a constant column (default True).
        verbose : bool/int, optional
            Print status during estimation. If an int is provided, prints
            status every verbose epochs.
        '''
        num_obs = len(X)
        if Y.ndim == 1:
            Y = np.reshape(Y, (-1, 1))
        if add_constant:
            X = np.column_stack([np.ones(num_obs), X])

        for i in range(epochnum):
            order = self._random.choice(num_obs, size=num_obs, replace=False)
            for ind in order:
                error = 0.

                curr_learn_rate = (self.learn_rate_evol(self.epochs_estimated)*
                                   self.learn_rate)

                curr_err = self._activate_backpropogate(
                    X, Y, ind, curr_learn_rate)

                error += np.mean(np.abs(curr_err))
            self.epochs_estimated += 1
            if verbose and (i % verbose) == (verbose-1):
                print('{:>4}, error: {:.3e}'.format(i+1, error/num_obs))

        return self

    def _activate_backpropogate(self, X, Y, ind, curr_learn_rate):
        targets = Y[ind]
        inputs = X[ind]

        # Update activations
        pred = inputs
        for l in self.layers:
            pred = l.update_activations(pred)

        # Backpropogate errors
        errors = targets - pred
        for l in reversed(self.layers):
            errors = l.backpropogate(inputs, errors, curr_learn_rate,
                                     self.momentum)

        return targets - pred


    def _predict_raw(self, X, add_constant=True):
        if add_constant:
            X = np.column_stack([np.ones(len(X)), X])

        pred = X
        for l in self.layers:
            pred = l.update_activations(pred)

        return pred

    def predict_prob(self, X, add_constant=True):
        raw = self._predict_raw(X, add_constant=add_constant)

        if raw.shape[1] == 1:
            return raw
        else:
            return raw / np.sum(raw, axis=1)[:, np.newaxis]

    def classify(self, X, add_constant=True, max_ind=False):
        raw = self._predict_raw(X, add_constant=add_constant)
        if max_ind:
            return np.argmax(raw, axis=1)
        else:
            return (raw > .5).astype(int)

