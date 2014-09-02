'''
Ensemble methods
'''
import numpy as np

from . import baseclasses as bc


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

class BaggingBinaryClassifier(EnsembleBinaryClassifier):
    _bag_params_names = ('model_params', 'n_models_fit', 'seed')

    def __init__(self, binaryclassifiercls, model_params=None, n_models_fit=10,
                 seed=None):
        super(type(self), self).__init__()

        self.base_model = binaryclassifiercls
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.n_models_fit = n_models_fit
        self.seed = seed

        self.obs_used = []
        self._oob_error = None

    @property
    def params(self):
        result = {}
        for name in self._bag_params_names:
            result[name] = getattr(self, name)
        return result

    @property
    def oob_error(self):
        if self._oob_error is None:
            raise AttributeError('Model has not been fitted yet.')
        return self._oob_error

    def fit(self, X, Y):
        '''
        Fit the bagged classifier using training data and calculates the
        out-of-bag fitting error.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature dataset for training.
        Y : array of shape (n_samples)
            Labels for training.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        num_obs = len(X)
        oob_votes_num = np.zeros(len(Y))
        oob_votes_for = np.zeros(len(Y))
        for i in range(self.n_models_fit):
            curr_ind = np.random.choice(num_obs, num_obs)
            curr_X = X[curr_ind]
            curr_Y = Y[curr_ind]

            curr_model = self.base_model(**self.model_params)
            curr_model.fit(curr_X, curr_Y)
            self.add_model(curr_model)

            obs_used = np.unique(curr_ind)
            self.obs_used.append(obs_used)

            obs_not_used = np.setdiff1d(np.arange(num_obs), obs_used,
                                        assume_unique=True)
            oob_votes_num[obs_not_used] += 1
            oob_votes_for[obs_not_used] += curr_model.classify(X[obs_not_used])

        oob_obs = oob_votes_num > 0
        oob_mean_votes = (oob_votes_for[oob_obs]/oob_votes_num[oob_obs]) >= .5
        self._oob_error = np.mean(oob_mean_votes != Y[oob_obs])

        return self

