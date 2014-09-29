import numpy as np
from nose.tools import raises

from simpleml.neural import AutoEncoder
from .test_dectree import X_TRAIN


def check_ae_same_est(mlp0, mlp1):
    for l0, l1 in zip(mlp0.layers, mlp1.layers):
        assert np.allclose(l0.weights, l1.weights)
        assert np.allclose(l0.activations, l1.activations)


class TestAEBasics:
    def test_nodes(self):
        ae = AutoEncoder(num_inputs=5, num_comp=3)
        assert ae.layers[0].num_inputs == 6
        assert ae.layers[0].num_nodes == 4


class TestAEFit:
    def test_fit(self):
        x = np.array([[0.1, 0.1], [0, 0.1], [1.1, 0.9], [1, 1.1]])

        ae = AutoEncoder(num_inputs=2, num_comp=1, seed=323440)
        ae.fit(x, epochnum=20)

class TestAEPredict:
    def setup(self):
        self.ae = AutoEncoder(num_inputs=3, num_comp=2, seed=323440)
        self.ae.fit(X_TRAIN, epochnum=20)

    @raises(AttributeError)
    def test_predict_prob(self):
        self.ae.predict_prob(X_TRAIN)

    @raises(AttributeError)
    def test_classify(self):
        self.ae.classify(X_TRAIN)

    def test_project(self):
        project = self.ae.project(X_TRAIN)
        assert project.shape == X_TRAIN.shape

    def test_transform(self):
        transform = self.ae.transform(X_TRAIN)
        assert transform.shape[1] == 2


class TestAESeed:
    params = {'num_inputs': 3, 'num_comp': 2}

    def test_same_seed(self):
        self.ae0 = AutoEncoder(seed=2345, **self.params)
        self.ae1 = AutoEncoder(seed=2345, **self.params)
        self.ae0.fit(X_TRAIN, epochnum=3)
        self.ae1.fit(X_TRAIN, epochnum=3)
        check_ae_same_est(self.ae0, self.ae1)

    def test_same_state(self):
        self.ae0 = AutoEncoder(seed=np.random.RandomState(345), **self.params)
        self.ae1 = AutoEncoder(seed=np.random.RandomState(345), **self.params)
        self.ae0.fit(X_TRAIN, epochnum=3)
        self.ae1.fit(X_TRAIN, epochnum=3)
        check_ae_same_est(self.ae0, self.ae1)

    def test_seed_persistence(self):
        self.ae0 = AutoEncoder(seed=2345, **self.params)
        self.ae1 = AutoEncoder(seed=2345, **self.params)
        self.ae0.fit(X_TRAIN, epochnum=6)
        self.ae1.fit(X_TRAIN, epochnum=3)
        np.random.normal(size=50)
        self.ae1.fit(X_TRAIN, epochnum=3)

        check_ae_same_est(self.ae0, self.ae1)
