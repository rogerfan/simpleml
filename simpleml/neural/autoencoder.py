'''
Auto-encoder.
'''
import numpy as np

from . import MultilayerPerceptron
from .. import metrics

__all__ = ('AutoEncoder',)


class AutoEncoder(MultilayerPerceptron):
    '''
    Auto-encoder.
    '''
    params_names = (
        'num_inputs', 'num_comp', 'learn_rate', 'learn_rate_evol',
        'momentum', 'seed', 'sigmoid', 'weight_init_range',
        'sparse', 'sparsity_target', 'sparsity_weight',
    )
    def __init__(self, num_inputs, num_comp=5,
                 learn_rate=.5, learn_rate_evol='linear', momentum=.1,
                 seed=None, sigmoid=metrics.logistic, weight_init_range=.5,
                 sparse=False, sparsity_target=.05, sparsity_weight=.2):
        super().__init__(
            num_inputs=num_inputs+1, num_outputs=num_inputs,
            num_hidden_layers=1, num_hidden_nodes=num_comp+1,
            learn_rate=learn_rate, learn_rate_evol=learn_rate_evol,
            momentum=.1, seed=seed, sigmoid=sigmoid,
            weight_init_range=weight_init_range)

        self.num_comp = num_comp
        self.sparse = sparse
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    @property
    def predict_prob(self, *pars, **npars):
        raise AttributeError(
            "'AutoEncoder' object has no attribute 'predict_prob'")

    @property
    def classify(self, *pars, **npars):
        raise AttributeError(
            "'AutoEncoder' object has no attribute 'classify'")

    def fit(self, X, epochnum=100, verbose=False):
        super().fit(X, X, epochnum=epochnum, add_constant=True,
                    verbose=verbose)

        return self

    def _activate_backpropogate(self, X, Y, ind, curr_learn_rate):
        targets = Y[ind]
        inputs = X[ind]

        # Calculate sparsity penalty
        if self.sparse:
            self.layers[0].update_activations(X)
            rhohat = np.mean(self.layers[0].activations, axis=0)
            penalty = (self.sparsity_weight * (
                -(self.sparsity_target / rhohat)  +
                (1 - self.sparsity_target) / (1 - rhohat)
            ))
        else:
            penalty = 0

        # Update activations
        pred = self._activate(inputs)

        # Backpropogate errors
        errors = self.layers[1].backpropogate(inputs, targets - pred,
                                              curr_learn_rate, self.momentum)
        self.layers[0].backpropogate(inputs, errors - penalty, curr_learn_rate,
                                     self.momentum)

        return targets - pred


    def project(self, X):
        return self._predict_raw(X, add_constant=True)

    def transform(self, X):
        self._predict_raw(X, add_constant=True)
        return self.layers[0].activations[:, 1:]

