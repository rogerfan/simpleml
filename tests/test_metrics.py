import numpy as np

import simpleml.metrics as met


PURITY_FUNCTION_LIST = [met.entropy, met.gini, met.misclass]


def create_funcs_generator(function_list):
    def test_funcs(self):
        for func in function_list:
            yield self.checker, func
    return test_funcs


class TestPurityHomogeneous:
    vals_to_test = (0., 1.)
    test_funcs = create_funcs_generator(PURITY_FUNCTION_LIST)

    def checker(self, func):
        for prop in self.vals_to_test:
            assert func(prop) == 0
            assert func(np.array([prop])) == 0

class TestPuritySymmetry:
    array_to_test = np.array([0., .1, .25, .45])
    test_funcs = create_funcs_generator(PURITY_FUNCTION_LIST)

    def checker(self, func):
        assert np.allclose(func(self.array_to_test), func(1-self.array_to_test))

class TestPurityDeriv:
    vals_to_test = [0., .05, 0.12, 0.18, 0.19, 0.20, 0.3, 0.41, 0.48, 0.49]
    test_funcs = create_funcs_generator(PURITY_FUNCTION_LIST)

    def checker(self, func):
        for a, b in zip(self.vals_to_test[:-1], self.vals_to_test[1:]):
            assert func(a) < func(b)

class TestPurityLargeVector:
    test_funcs = create_funcs_generator(PURITY_FUNCTION_LIST)

    def setup(self):
        np.random.seed(1254)
        self.array_to_test = np.random.randint(0, 1, size=5000)

    def checker(self, func):
        func(self.array_to_test)
