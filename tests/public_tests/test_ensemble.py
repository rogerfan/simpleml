import numpy as np
from nose.tools import raises

import simpleml.ensemble as ens

# Simple classifiers for testing
class NotBinaryClassifier:  # pragma: no cover
    def fit(self):
        pass
    def test_err(self):
        pass

class ExBinaryClassifier1(NotBinaryClassifier):  # pragma: no cover
    answer = 1
    def classify(self, X):
        if len(X.shape) == 1:
            return np.array(self.answer)
        else:
            return np.ones(len(X)) * self.answer

class ExBinaryClassifier0(ExBinaryClassifier1):  # pragma: no cover
    answer = 0

class SimpleClassifier:  # pragma: no cover
    def __init__(self, ind=0):
        self.ind = ind
        self.train_err = None

    def fit(self, X, Y):
        self.param = (np.round(Y[X[:, self.ind] < 0].mean()),
                      np.round(Y[X[:, self.ind] >= 0].mean()))

    def classify(self, X):
        return np.where(X[:, self.ind] < 0,
                        np.ones(len(X))*self.param[0],
                        np.ones(len(X))*self.param[1])

    def test_err(self, X, Y):
        return np.mean(self.classify(X) != Y)


class TestEnsembleBinaryClassifierAdd:
    num_model = 5

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()

    @raises(TypeError)
    def test_not_binary_classifier(self):
        self.ensemble.add_model(NotBinaryClassifier())

    def test_add_model(self):
        for i in range(self.num_model):
            self.ensemble.add_model(ExBinaryClassifier1())

        assert len(self.ensemble.models) == self.ensemble.n_models
        assert len(self.ensemble.weights) == self.ensemble.n_models
        assert self.ensemble.n_models == self.num_model
        for weight in self.ensemble.weights:
            assert weight == 1

    def test_add_model_weights(self):
        for i in range(self.num_model):
            self.ensemble.add_model(ExBinaryClassifier1(), weight=i)

        for weight, i in zip(self.ensemble.weights, range(self.num_model)):
            assert weight == i


class TestEnsembleBinaryClassifier1:
    answer = 1
    X = np.array([[1, 3, 5],
                  [2, 4, 6],
                  [1, 3, 6]])

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(ExBinaryClassifier1())
        self.ensemble.add_model(ExBinaryClassifier1())
        self.ensemble.add_model(ExBinaryClassifier0())

    def test_classify(self):
        res = self.ensemble.classify(self.X)
        assert np.allclose(res, np.ones(len(self.X))*self.answer)

    def test_classify_1obs(self):
        res = self.ensemble.classify(np.array([1, 3, 5]))
        assert res == np.ones(1)*self.answer

class TestEnsembleBinaryClassifier0(TestEnsembleBinaryClassifier1):
    answer = 0

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(ExBinaryClassifier1())
        self.ensemble.add_model(ExBinaryClassifier0())
        self.ensemble.add_model(ExBinaryClassifier0())

class TestEnsembleBinaryClassifierWeight1(TestEnsembleBinaryClassifier1):
    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(ExBinaryClassifier1(), weight=2)
        self.ensemble.add_model(ExBinaryClassifier0(), weight=1)

class TestEnsembleBinaryClassifierWeight0(TestEnsembleBinaryClassifier0):
    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(ExBinaryClassifier1(), weight=1)
        self.ensemble.add_model(ExBinaryClassifier0(), weight=2)


class TestBaggingBinaryClassifierInit:
    def test_init1(self):
        model_params = {'a': 5, 'b': True}
        bag_params = {'model_params': model_params, 'n_models_fit': 5,
                      'seed': 23}
        bag = ens.BaggingBinaryClassifier(ExBinaryClassifier1,
                                          model_params=model_params,
                                          n_models_fit=5, seed=23)
        bag.params == bag_params

    def test_init2(self):
        bag_params = {'model_params': {}, 'n_models_fit': 5, 'seed': None}
        bag = ens.BaggingBinaryClassifier(ExBinaryClassifier1, n_models_fit=5)
        bag.params == bag_params


class TestBaggingBinaryClassifierOobErrors:
    X = np.array([[ 1,-1],
                  [-1, 1],
                  [ 1,-1],
                  [ 1, 1],
                  [-1, 1]])
    Y = np.array([1, 0, 1, 0, 0])

    @raises(AttributeError)
    def test_oob_nofit(self):
        bag = ens.BaggingBinaryClassifier(ExBinaryClassifier1)
        bag.oob_error

    @raises(TypeError)
    def test_fit_oob_noerror(self):
        bag = ens.BaggingBinaryClassifier(ExBinaryClassifier1)
        bag.fit(self.X, self.Y, oob_error=True)


class TestBaggingBinaryClassifier:
    X = np.array([[ 1,-1],
                  [-1, 1],
                  [ 1,-1],
                  [ 1, 1],
                  [-1, 1]])
    Y = np.array([1, 0, 1, 0, 0])

    def test_fit(self):
        bag_params = {'model_params': {'ind': 0}, 'n_models_fit': 5, 'seed': 23}
        bag = ens.BaggingBinaryClassifier(SimpleClassifier, **bag_params)
        bag.fit(self.X, self.Y, oob_error=False)
        assert bag.params == bag_params
        assert bag.n_models == bag_params['n_models_fit'] == bag.n_models_fit

    def test_fit_oob(self):
        bag = ens.BaggingBinaryClassifier(SimpleClassifier)
        bag.fit(self.X, self.Y, oob_error=True)
        assert np.isfinite(bag.oob_error)
        assert 0. <= bag.oob_error <= 1.

    def test_classify_easy(self):
        bag_params = {'model_params': {'ind': 1}, 'n_models_fit': 10,
                      'seed': 23}
        bag = ens.BaggingBinaryClassifier(SimpleClassifier,
                                          **bag_params)
        bag.fit(self.X, self.Y)
        assert np.all(bag.classify(self.X) == self.Y)
