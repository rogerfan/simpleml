'''
Random forests.
'''
from .bagging import BaggingBinaryClassifier
from .. import metrics
from ..dectree import DecisionTree


class RandomForest(BaggingBinaryClassifier):
    '''
    Random forest classifier.

    Parameters
    ----------
    min_obs_split : int [2]
        Nodes with sizes less than this will not be split further.
    max_depth : int [None]
        Maximum depth to grow the tree to.
    objfunc : function [metrics.gini]
        Objective function to minimize when selecting splits.
    max_features : int [10]
        Number of features to randomly choose to consider at each split point.
        The square root of the number of features is often used as a guideline
        for what to set this to.
    n_models_fit : int [10]
        Number of bagging models to fit.
    seed : int [None]
        If provided, seeds the random number generator.
    '''
    def __init__(self, min_obs_split=2, max_depth=None, objfunc=metrics.gini,
                 max_features=10, n_models_fit=10, seed=None):
        model_params = {
            'min_obs_split': min_obs_split,
            'max_depth': max_depth,
            'objfunc': objfunc,
            'max_features': max_features
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
