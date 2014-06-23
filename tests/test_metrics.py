import numpy as np

import simpleml.metrics as met


class TestPurityHomogeneous:
    def setup(self):
        self.all0 = np.array([0 for i in range(5)])
        self.all1 = np.array([1 for i in range(5)])

    def test_entropy(self):
        assert met.entropy(self.all0) == 0
        assert met.entropy(self.all1) == 0

    def test_gini(self):
        assert met.gini(self.all0) == 0
        assert met.gini(self.all1) == 0

    def test_misclass(self):
        assert met.misclass(self.all0) == 0
        assert met.misclass(self.all1) == 0

class TestPuritySymmetry:
    def setup(self):
        self.labels0 = np.array([0 for i in range(4)] + [1])
        self.labels1 = np.array([1 for i in range(4)] + [0])

    def test_entropy(self):
        assert np.isclose(met.entropy(self.labels0), met.entropy(self.labels1))

    def test_gini(self):
        assert np.isclose(met.gini(self.labels0), met.gini(self.labels1))

    def test_misclass(self):
        assert np.isclose(met.misclass(self.labels0), met.misclass(self.labels1))

class TestPurityDeriv:
    def setup(self):
        self.labels = [
            np.array([0, 0, 0, 0, 0]),
            np.array([1, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0])
        ]

    def test_entropy(self):
        assert met.entropy(self.labels[0]) < met.entropy(self.labels[1])
        assert met.entropy(self.labels[1]) < met.entropy(self.labels[2])

    def test_gini(self):
        assert met.gini(self.labels[0]) < met.gini(self.labels[1])
        assert met.gini(self.labels[1]) < met.gini(self.labels[2])

    def test_misclass(self):
        assert met.misclass(self.labels[0]) < met.misclass(self.labels[1])
        assert met.misclass(self.labels[1]) < met.misclass(self.labels[2])
