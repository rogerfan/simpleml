import numpy as np
from nose.tools import raises

import simpleml.ensemble as ens


X = np.array([[1, 3, 5],
              [2, 4, 6],
              [1, 3, 6]])


class NotBinaryClassifier:  # pragma: no cover
    def fit(self):
        pass

class SimpleBinaryClassifier1(NotBinaryClassifier):  # pragma: no cover
    answer = 1
    def classify(self, X):
        if len(X.shape) == 1:
            return np.array(self.answer)
        else:
            return np.ones(len(X)) * self.answer

class SimpleBinaryClassifier0(SimpleBinaryClassifier1):  # pragma: no cover
    answer = 0


class TestEnsembleBinaryClassifierAdd:
    num_model = 5

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()

    @raises(TypeError)
    def test_not_binary_classifier(self):
        self.ensemble.add_model(NotBinaryClassifier())

    def test_add_model(self):
        for i in range(self.num_model):
            self.ensemble.add_model(SimpleBinaryClassifier1())

        assert len(self.ensemble.models) == self.ensemble.n_models
        assert len(self.ensemble.weights) == self.ensemble.n_models
        assert self.ensemble.n_models == self.num_model
        for weight in self.ensemble.weights:
            assert weight == 1

    def test_add_model_weights(self):
        for i in range(self.num_model):
            self.ensemble.add_model(SimpleBinaryClassifier1(), weight=i)

        for weight, i in zip(self.ensemble.weights, range(self.num_model)):
            assert weight == i


class TestEnsembleBinaryClassifier1:
    answer = 1

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(SimpleBinaryClassifier1())
        self.ensemble.add_model(SimpleBinaryClassifier1())
        self.ensemble.add_model(SimpleBinaryClassifier0())

    def test_classify(self):
        res = self.ensemble.classify(X)
        assert np.allclose(res, np.ones(len(X))*self.answer)

    def test_classify_1obs(self):
        res = self.ensemble.classify(np.array([1, 3, 5]))
        assert res == np.ones(1)*self.answer

class TestEnsembleBinaryClassifier0(TestEnsembleBinaryClassifier1):
    answer = 0

    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(SimpleBinaryClassifier1())
        self.ensemble.add_model(SimpleBinaryClassifier0())
        self.ensemble.add_model(SimpleBinaryClassifier0())

class TestEnsembleBinaryClassifierWeight1(TestEnsembleBinaryClassifier1):
    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(SimpleBinaryClassifier1(), weight=2)
        self.ensemble.add_model(SimpleBinaryClassifier0(), weight=1)

class TestEnsembleBinaryClassifierWeight0(TestEnsembleBinaryClassifier0):
    def setup(self):
        self.ensemble = ens.EnsembleBinaryClassifier()
        self.ensemble.add_model(SimpleBinaryClassifier1(), weight=1)
        self.ensemble.add_model(SimpleBinaryClassifier0(), weight=2)
