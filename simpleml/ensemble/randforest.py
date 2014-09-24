'''
Random forests.
'''
import numpy as np

from .bagging import BaggingBinaryClassifier
from .. import metrics
from ..dectree import DecisionTree

__all__ = ('RandomForest',)


class RandomForest(BaggingBinaryClassifier):
    '''
    Random forest classifier.

    Parameters
    ----------
    min_obs_split : int, optional
        Nodes with sizes less than this will not be split further (default 2).
    max_depth : int, optional
        Maximum depth to grow the tree to (default None).
    objfunc : function, optional
        Objective function to minimize when selecting splits
        (default metrics.gini).
    max_features : int, optional
        Number of features to randomly choose to consider at each split point.
        The square root of the number of features is often used as a guideline
        for what to set this to (default 10).
    n_models_fit : int, optional
        Number of bagging models to fit (default 10).
    seed : int or RandomState, optional
        If provided, seeds the random number generator (default None).
    '''
    def __init__(self, min_obs_split=2, max_depth=None, objfunc=metrics.gini,
                 max_features=10, n_models_fit=10, seed=None):
        if max_features <= 0:
            raise ValueError('max_features must be positive.')
        if isinstance(seed, int):
            modelseed = np.random.RandomState(seed)
        else:
            modelseed = seed

        model_params = {
            'min_obs_split': min_obs_split,
            'max_depth': max_depth,
            'objfunc': objfunc,
            'max_features': max_features,
            'seed': modelseed
        }

        super().__init__(
            DecisionTree, model_params=model_params, n_models_fit=n_models_fit,
            seed=seed
        )

    @property
    def params(self):
        result = {}
        for name in self._bag_params_names:
            if name != 'model_params':
                result[name] = getattr(self, name)
        for key, val in self.model_params.items():
            result[key] = val
        return result
