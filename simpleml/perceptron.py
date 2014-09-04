'''
Multi-layer perceptrons.
'''
import numpy as np

from . import metrics


class _Layer:
    def __init__(self, num_inputs, num_nodes, parent=None,
                 sigmoid=metrics.tanh, weight_init_range=.5):
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

    def __repr__(self):
        return "{}\n{}".format(self.activations, self.weights)

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
