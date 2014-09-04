'''
Base classes for ensemble estimators.
'''
import numpy as np

from .. import baseclasses as bc

__all__ = ()


class EnsembleBinaryClassifier:
    '''
    Collects multiple binary classifiers into a (weighted) voting ensemble
    classifier.

    Attributes
    ----------
    models : list
        Models in the ensemble.
    n_models : int
        Number of models in the ensemble.
    weights : list
        Relative weights for each model in the voting.
    '''
    def __init__(self):
        self.models = []
        self.n_models = 0
        self.weights = []

    def add_model(self, model, weight=1):
        '''
        Adds a model to the ensemble.

        Parameters
        ----------
        model : BinaryClassifier
            An object that satisifes the baseclasses.BinaryClassifier abstract
            base class. Must have methods 'fit', 'classify', and 'test_err'.
            Should already be fitted.
        weight : float [1]
            Relative weight when voting.
        '''
        typename = type(model).__name__
        if not isinstance(model, bc.BinaryClassifier):
            raise TypeError("Model is '{}', which is not a "
                            "'BinaryClassifier'.".format(typename))
        self.models.append(model)
        self.n_models += 1
        self.weights.append(weight)
        return self

    def classify(self, X):
        '''
        Classify data.

        Parameters
        ----------
        X : array of shape (n_features) or (n_samples, n_features)
            Feature data to classify. Can be a single or multiple observations.
        '''
        if len(X.shape) == 1:
            n_obs = 1
        else:
            n_obs = len(X)

        model_preds = np.zeros((n_obs, self.n_models))
        for i, model in enumerate(self.models):
            model_preds[:,i] = model.classify(X)
        results = np.average(model_preds, axis=1, weights=self.weights)
        return results >= .5

    def test_err(self, X, Y):
        '''
        Compute test error.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature dataset for testing.
        Y : array of shape (n_samples)
            Labels for testing.
        '''
        return np.mean(self.classify(X) != Y)
