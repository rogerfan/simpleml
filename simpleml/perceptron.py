'''
Multi-layer perceptrons.
'''
import numpy as np

from . import metrics


class Layer:
    def __init__(self, num_inp, num_nodes, parent=None, sigmoid=metrics.tanh):
        self.num_nodes = num_nodes
        self.num_inp = num_inp
        self.sigmoid = sigmoid

        self.parent = parent
        self.child = None
        if parent is not None:
            parent.child = self

        self.activations = np.zeros(num_nodes)
        self.weights = np.random.uniform(-.5, .5, (num_inp, num_nodes))
        self._last_change = np.zeros((num_inp, num_nodes))
        self._deltas = np.zeros(num_nodes)

    def __repr__(self):
        return "{}\n{}".format(self.activations, self.weights)

    def update_activations(self, inputs):
        self.activations = self.sigmoid.f(np.dot(inputs, self.weights))
        return self.activations

    def backpropogate(self, inputs, errors, learn_rate, momentum):
        self._deltas = self.sigmoid.d(self.activations)*errors
        self._last_change = (np.outer(inputs, self._deltas) +
                             momentum*self._last_change)
        self.weights += learn_rate*self._last_change

        return np.dot(self._deltas, self.weights.T)
