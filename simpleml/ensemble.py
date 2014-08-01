'''
Ensemble methods
'''
import numpy as np

from . import baseclasses as bc


class EnsembleBinaryClassifier:
    def __init__(self):
        self.models = []
        self.n_models = 0
        self.weights = []

    def add_model(self, model, weight=1):
        typename = type(model).__name__
        if not isinstance(model, bc.BinaryClassifier):
            raise TypeError("Model is '{}', which is not a "
                            "'BinaryClassifier'.".format(typename))
        self.models.append(model)
        self.n_models += 1
        self.weights.append(weight)

    def classify(self, X):
        if len(X.shape) == 1:
            n_obs = 1
        else:
            n_obs = len(X)

        model_preds = np.zeros((n_obs, self.n_models))
        for i, model in enumerate(self.models):
            model_preds[:,i] = model.classify(X)
        results = np.average(model_preds, axis=1, weights=self.weights)
        return results >= .5


class BaggingBinaryClassifier(EnsembleBinaryClassifier):
    bag_params_names = ('model_params', 'n_models', 'seed')

    def __init__(self, binaryclassifiercls, model_params=None, n_models=10,
                 seed=None):
        super(BaggingBinaryClassifier, self).__init__()

        self.base_model = binaryclassifiercls
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.n_models = n_models
        self.seed = seed

        self.obs_used = []
        self._oob_error = None
        self._oob_num = 0

    @property
    def params(self):
        result = {}
        for name in self.bag_params_names:
            result[name] = getattr(self, name)
        return result

    @property
    def oob_error(self):
        if not issubclass(self.base_model, bc.BinaryClassifierWithErrors):
            raise TypeError("Base model is '{}', which is not a "
                            "'BinaryClassifierWithErrors'"
                            ".".format(self.base_model.__name__))
        if self._oob_error is None:
            raise AttributeError('Model has not been fitted yet.')
        return self._oob_error

    def fit(self, X, Y, oob_error=True):
        if oob_error:
            if not issubclass(self.base_model, bc.BinaryClassifierWithErrors):
                raise TypeError("Base model is '{}', which is not a "
                                "'BinaryClassifierWithErrors'"
                                ".".format(self.base_model.__name__))
            self._oob_error = 0

        num_obs = len(X)
        for i in range(self.n_models):
            curr_ind = np.random.choice(num_obs, num_obs)
            curr_X = X[curr_ind]
            curr_Y = Y[curr_ind]

            curr_model = self.base_model(**self.model_params)
            curr_model.fit(curr_X, curr_Y)

            obs_used = np.unique(curr_ind)

            self.add_model(curr_model)
            self.obs_used.append(obs_used)

            # obs_not_used = np.setdiff1d(np.arange(num_obs), obs_used,
            #                             assume_unique=True)

        return self

