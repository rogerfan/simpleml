'''
Bagging meta-classifiers.
'''
import sys

import numpy as np

from .base import EnsembleBinaryClassifier

__all__ = ('BaggingBinaryClassifier',)


class BaggingBinaryClassifier(EnsembleBinaryClassifier):
    '''
    Bagging binary classifier.

    Parameters
    ----------
    binaryclassifiercls : object
        Binary classifier used to create the bagging classifier.
    model_params : dict, optional
        Dictionary of named arguments that are used by binaryclassifiercls
        (default {}).
    n_models_fit : int, optional
        Number of bagging estimators to fit (default 10).
    seed : int, optional
        If provided, seeds the random number generator (default 10).
    '''
    _bag_params_names = ('model_params', 'n_models_fit', 'seed')

    def __init__(self, binaryclassifiercls, model_params=None, n_models_fit=10,
                 seed=None):
        super().__init__()

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

    def fit(self, X, Y, fit_params=None, verbose=False):
        '''
        Fit the bagged classifier using training data and calculates the
        out-of-bag fitting error.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature dataset for training.
        Y : array of shape (n_samples)
            Labels for training.
        fit_params : dict, optional
            Dictionary of named arguments that are used by
            binaryclassifiercls.fit (default {}).
        verbose : bool, optional
            Print status during estimation.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        if fit_params is None:
            fit_params = {}

        num_obs = len(X)
        oob_votes_num = np.zeros(len(Y))
        oob_votes_for = np.zeros(len(Y))
        for i in range(self.n_models_fit):
            curr_ind = np.random.choice(num_obs, num_obs)
            curr_X = X[curr_ind]
            curr_Y = Y[curr_ind]

            curr_model = self.base_model(**self.model_params)
            curr_model.fit(curr_X, curr_Y, **fit_params)
            self.add_model(curr_model)

            obs_used = np.unique(curr_ind)
            self.obs_used.append(obs_used)

            obs_not_used = np.setdiff1d(np.arange(num_obs), obs_used,
                                        assume_unique=True)
            oob_votes_num[obs_not_used] += 1
            oob_votes_for[obs_not_used] += curr_model.classify(X[obs_not_used])

            if verbose:
                sys.stdout.write(".")
                if (i % 10) == 9:
                    sys.stdout.write("\n")
                sys.stdout.flush()

        oob_obs = oob_votes_num > 0
        oob_mean_votes = (oob_votes_for[oob_obs]/oob_votes_num[oob_obs]) >= .5
        self._oob_error = np.mean(oob_mean_votes != Y[oob_obs])

        return self

