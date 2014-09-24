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
        'momentum', 'seed', 'sigmoid', 'weight_init_range'
    )
    def __init__(self, num_inputs, num_comp=5,
                 learn_rate=.5, learn_rate_evol='linear', momentum=.1,
                 seed=None, sigmoid=metrics.logistic, weight_init_range=.5):
        super().__init__(
            num_inputs=num_inputs+1, num_outputs=num_inputs,
            num_hidden_layers=1, num_hidden_nodes=num_comp+1,
            learn_rate=learn_rate, learn_rate_evol=learn_rate_evol,
            momentum=.1, seed=seed, sigmoid=sigmoid,
            weight_init_range=weight_init_range)

        self.num_comp = num_comp

    @property
    def predict_prob(self, *pars, **npars):
        raise AttributeError(
            "'AutoEncoder' object has no attribute 'predict_prob'")

    @property
    def classify(self, *pars, **npars):
        raise AttributeError(
            "'AutoEncoder' object has no attribute 'classify'")

    def fit(self, X, epochnum=100, add_constant=True, verbose=False):
        super().fit(X, X, epochnum=epochnum, add_constant=add_constant,
                    verbose=verbose)

        return self

    def project(self, X):
        pass

    def transform(self, X):
        pass

