import numpy as np
from nose.tools import raises

from simpleml.neural import MultilayerPerceptron
import simpleml.metrics as metrics
from simpleml.transform import to_dummies
from .test_dectree import X_TRAIN, LABELS_TRAIN


def check_mlp_same_est(mlp0, mlp1):
    for l0, l1 in zip(mlp0.layers, mlp1.layers):
        assert np.allclose(l0.weights, l1.weights)
        assert np.allclose(l0.activations, l1.activations)


def create_learn_rate_funcs_gen(testcases):
    def test_values(case):
        mlp = MultilayerPerceptron(learn_rate_evol=case[0])
        for arg, out in case[1]:
            assert np.isclose(mlp.learn_rate_evol(arg), out)

    def test_learn_rates(self):
        for case in testcases:
            yield test_values, case

    return test_learn_rates

class TestMLPLearnRate:
    cases = [
        ('constant', [(0, 1.), (1, 1.), (2, 1.), (9, 1.)]),
        ('linear', [(0, 1.), (1, 1/2), (2, 1/3), (9, 1/10)]),
        ('quadratic', [(0, 1.), (1, 1/4), (2, 1/9), (9, 1/100)]),
        (lambda a: 1 / (a+1)**3, [(0, 1.), (1, 1/8), (2, 1/27), (9, 1/1000)]),
    ]

    test_learn_rates = create_learn_rate_funcs_gen(cases)

    @raises(ValueError)
    def test_not_recognized(self):
        MultilayerPerceptron(learn_rate_evol='unrecognized')

class TestMLPBasics:
    def test_params(self):
        param_dict = {
            'num_inputs': 4, 'num_outputs': 2,
            'num_hidden_layers': 2, 'num_hidden_nodes': [2, 3],
            'learn_rate': .4, 'momentum': .2, 'seed': 24545,
            'sigmoid': metrics.tanh, 'weight_init_range': .2
        }

        mlp = MultilayerPerceptron(**param_dict)
        for key, val in param_dict.items():
            assert mlp.params[key] == val

    def test_nodes(self):
        mlp = MultilayerPerceptron(num_hidden_layers=3, num_hidden_nodes=3)
        assert mlp.params['num_hidden_nodes'] == [3, 3, 3]

    @raises(ValueError)
    def test_nodes_wrong_number(self):
        MultilayerPerceptron(num_hidden_layers=1, num_hidden_nodes=[3, 3])

    def test_print(self):
        mlp = MultilayerPerceptron()
        print(mlp)

    def test_copy(self):
        mlp0 = MultilayerPerceptron()
        mlp1 = mlp0.copy()

        assert mlp0.params == mlp1.params
        check_mlp_same_est(mlp0, mlp1)
        for l0, l1 in zip(mlp0.layers, mlp1.layers):
            assert not np.may_share_memory(l0.weights, l1.weights)
            assert not np.may_share_memory(l0.activations, l1.activations)
            assert l0.bias == l1.bias

class TestMLPFit:
    def test_easy(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 0, 1])

        mlp = MultilayerPerceptron(
            num_inputs=3, num_hidden_layers=1, num_hidden_nodes=6, seed=323440,
            learn_rate = .8, learn_rate_evol='constant', momentum=.1
        )
        mlp.fit(x, y, add_constant=True, epochnum=100, verbose=True)

        print(mlp.classify(x, add_constant=True))
        assert np.allclose(mlp.classify(x, add_constant=True).reshape(len(y)),
                           y)

    def test_easy_multidim_y(self):
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])

        mlp = MultilayerPerceptron(
            num_inputs=3, num_outputs=2,
            num_hidden_layers=1, num_hidden_nodes=6, seed=323440,
            learn_rate = .8, learn_rate_evol='constant', momentum=.1
        )
        mlp.fit(x, y, epochnum=50)
        results = mlp.classify(x, max_ind=True)
        assert np.allclose(to_dummies(results), y)

    def test_fit(self):
        x = np.column_stack([np.ones(len(X_TRAIN)), X_TRAIN])
        mlp = MultilayerPerceptron(
            num_inputs=4, num_hidden_layers=1, num_hidden_nodes=3)
        mlp.fit(x, LABELS_TRAIN, epochnum=5, add_constant=False)
        mlp.classify(x, add_constant=False)


class TestMLPSeed:
    params = {'num_inputs':4, 'num_hidden_layers':1, 'num_hidden_nodes':3}

    def test_same_seed(self):
        self.mlp0 = MultilayerPerceptron(seed=2345, **self.params)
        self.mlp1 = MultilayerPerceptron(seed=2345, **self.params)
        self.mlp0.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)
        self.mlp1.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)
        check_mlp_same_est(self.mlp0, self.mlp1)

    def test_same_state(self):
        self.mlp0 = MultilayerPerceptron(seed=np.random.RandomState(345),
                                         **self.params)
        self.mlp1 = MultilayerPerceptron(seed=np.random.RandomState(345),
                                         **self.params)
        self.mlp0.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)
        self.mlp1.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)
        check_mlp_same_est(self.mlp0, self.mlp1)

    def test_seed_persistence(self):
        self.mlp0 = MultilayerPerceptron(seed=2345, **self.params)
        self.mlp1 = MultilayerPerceptron(seed=2345, **self.params)
        self.mlp0.fit(X_TRAIN, LABELS_TRAIN, epochnum=6)
        self.mlp1.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)
        np.random.normal(size=50)
        self.mlp1.fit(X_TRAIN, LABELS_TRAIN, epochnum=3)

        check_mlp_same_est(self.mlp0, self.mlp1)
