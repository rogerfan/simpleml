import numpy as np
from nose.tools import raises

from simpleml.perceptron import MultilayerPerceptron
import simpleml.metrics as metrics
from simpleml.transform import to_dummies
from .test_dectree import X_TRAIN, LABELS_TRAIN


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
        mlp = MultilayerPerceptron()
        mlp2 = mlp.copy()

        assert mlp.params == mlp2.params
        for l1, l2 in zip(mlp.layers, mlp2.layers):
            assert np.allclose(l1.weights, l2.weights)
            assert not np.may_share_memory(l1.weights, l2.weights)
            assert np.allclose(l1.activations, l2.activations)
            assert not np.may_share_memory(l1.activations, l2.activations)
            assert l1.bias == l2.bias

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

