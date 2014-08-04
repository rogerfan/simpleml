import numpy as np
from nose.tools import raises

import simpleml.dectree as dt
import simpleml.metrics as metrics


X_TRAIN = np.array([
    [-1.4273, -1.0824,  0.0058],
    [ 0.7866, -0.3452, -0.5211],
    [ 1.1191,  0.0517, -1.1401],
    [ 1.6514,  0.2658,  0.4943],
    [ 0.3742, -0.9247,  0.1196],
    [-0.5938, -0.9151, -0.2263],
    [ 1.0806, -0.5539, -1.9748],
    [-0.2261,  0.228 ,  0.5221],
    [ 1.0467,  0.9656,  0.4729],
    [-1.327 , -0.7972, -1.4782],
    [ 0.6219, -0.0472, -1.178 ],
    [ 0.4702,  0.16  , -0.111 ],
    [-0.7008, -0.586 , -1.2022],
    [-0.0925,  0.5194,  1.7272],
    [ 1.6551, -0.3479, -2.0299],
    [-1.0322, -0.6711, -1.476 ],
    [ 0.6841, -1.2086, -1.6619],
    [ 0.3332, -1.8284, -0.291 ],
    [-1.8365,  1.2062, -0.888 ],
    [-0.2278, -1.2292,  1.5369]
])

LABELS_TRAIN = np.array(
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
)

X_TEST = np.array([
    [-0.4278, -0.9836, -0.0415],
    [-0.6813, -0.1991, -2.1691],
    [-1.5937, -0.9405, -1.6823],
    [ 1.8772,  1.0064,  0.0475],
    [ 0.736 ,  1.0728, -0.506 ],
    [-0.3631, -0.6367,  0.7501],
    [ 1.6533,  0.3377, -0.6203],
    [ 0.1004,  0.522 ,  0.131 ],
    [-1.1707,  2.5464,  0.4533],
    [ 0.6375, -1.2981,  0.2451],
    [ 0.1535,  0.8227,  0.9552],
    [ 0.6844, -0.468 , -0.989 ],
    [ 0.893 ,  2.8725,  0.5143],
    [ 0.2305,  0.5278,  0.2616],
    [ 0.7665, -0.4036,  0.8091],
    [ 0.0776,  0.5119,  0.1585],
    [-0.095 ,  1.2517,  0.3549],
    [-0.0125,  0.6841,  2.038 ],
    [-1.5437, -1.225 ,  0.1346],
    [ 0.4143,  0.2247, -0.0301]
])

LABELS_TEST = np.array(
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
)

LABELS_TRAIN_BOOL = (LABELS_TRAIN == 1)
LABELS_TEST_BOOL = (LABELS_TEST == 1)


class TestDecisionTreeInit:
    def test_params(self):
        param_dict = {
            'min_obs_split': 4,
            'max_depth': 2,
            'objfunc': metrics.gini,
            'max_features': 2,
            'seed': 153
        }

        dtree = dt.DecisionTree(**param_dict)
        for key, val in param_dict.items():
            assert dtree.params[key] == val

    def test_params_empty(self):
        param_dict = {
            'min_obs_split': None,
            'max_depth': None,
            'objfunc': metrics.gini,
            'max_features': None,
            'seed': None
        }

        dtree = dt.DecisionTree(**param_dict)
        for key, val in param_dict.items():
            if key == 'max_depth':
                assert dtree.params[key] == np.inf
            else:
                assert dtree.params[key] == val


class TestDecisionTreeFit:
    labels_train = LABELS_TRAIN

    def test_fit(self):
        dtree = dt.DecisionTree()
        dtree.fit(X_TRAIN, self.labels_train)
        assert dtree.tree is not None
        assert dtree.train_err is not None

    def test_max_features_inrange(self):
        dtree = dt.DecisionTree(max_depth=1, max_features=2)
        dtree.fit(X_TRAIN, self.labels_train)

    @raises(ValueError)
    def test_max_features_outofrange1(self):
        dtree = dt.DecisionTree(max_depth=1, max_features=4)
        dtree.fit(X_TRAIN, self.labels_train)

    @raises(ValueError)
    def test_max_features_outofrange2(self):
        dtree = dt.DecisionTree(max_depth=1, max_features=-1)
        dtree.fit(X_TRAIN, self.labels_train)

    def test_seed(self):
        seed = 195
        dtree1 = dt.DecisionTree(max_features=1, seed=seed)
        dtree2 = dt.DecisionTree(max_features=1, seed=seed)
        dtree1.fit(X_TRAIN, self.labels_train)
        dtree2.fit(X_TRAIN, self.labels_train)

        for desc1, desc2 in zip(dtree1.tree.descendents(),
                                dtree2.tree.descendents()):
            assert desc1.split == desc2.split

    def test_copy(self):
        dtree1 = dt.DecisionTree()
        dtree1.fit(X_TRAIN, self.labels_train)
        dtree2 = dtree1.copy()

        assert dtree1.tree is not dtree2.tree
        for desc1, desc2 in zip(dtree1.tree.descendents(),
                                dtree2.tree.descendents()):
            assert desc1.split == desc2.split

    def test_classify(self):
        dtree = dt.DecisionTree()
        dtree.fit(X_TRAIN, self.labels_train)
        assert np.all(dtree.classify(X_TRAIN) == self.labels_train)
        assert np.all(dtree.classify(X_TRAIN[5]) == self.labels_train[5])

    def test_print(self):
        dtree = dt.DecisionTree()
        dtree.fit(X_TRAIN, self.labels_train)
        print(dtree)


class TestEasyDecisionTreeError:
    labels_train = LABELS_TRAIN
    labels_test = LABELS_TEST

    def setup(self):
        self.x_train =  X_TRAIN.copy()
        self.x_test =  X_TEST.copy()

        self.x_train[:,2] = np.abs(self.x_train[:,2]) * (-1+2*self.labels_train)
        self.x_test[:,2]  = np.abs(self.x_test[:,2])  * (-1+2*self.labels_test)

        self.dtree = dt.DecisionTree()
        self.dtree.fit(self.x_train, self.labels_train)

    def test_train_err(self):
        assert self.dtree.train_err == 0

    def test_test_err(self):
        assert self.dtree.test_err(self.x_test, self.labels_test) == 0

    def test_test_err_newdata(self):
        assert self.dtree.test_err(X_TRAIN, self.labels_train) < 1

class TestEasyDecisionTreeErrorBin(TestEasyDecisionTreeError):
    def setup(self):
        self.x_train =  X_TRAIN.copy()
        self.x_test =  X_TEST.copy()

        self.x_train[:,2] = -0.5 + self.labels_train
        self.x_test[:,2]  = -0.5 + self.labels_test

        self.dtree = dt.DecisionTree()
        self.dtree.fit(self.x_train, self.labels_train)

class TestEasyDecisionTreeErrorCat(TestEasyDecisionTreeError):
    def setup(self):
        self.x_train =  X_TRAIN.copy()
        self.x_test =  X_TEST.copy()

        self.x_train[:,2] = (-0.5 + self.labels_train +
                             np.logical_and(self.labels_train,
                                            self.x_train[:,2] > 0))
        self.x_test[:,2]  = (-0.5 + self.labels_test +
                             np.logical_and(self.labels_test,
                                            self.x_test[:,2] > 0))

        self.dtree = dt.DecisionTree()
        self.dtree.fit(self.x_train, self.labels_train)


class TestDecisionTreePrune:
    labels_train = LABELS_TRAIN
    labels_test = LABELS_TEST

    def setup(self):
        self.dtree = dt.DecisionTree()
        self.dtree.fit(X_TRAIN, self.labels_train)

    def test_prune_err(self):
        naive_test_err = self.dtree.test_err(X_TEST, self.labels_test)
        naive_train_err = self.dtree.train_err
        naive_num_nodes = self.dtree.tree.num_nodes()

        self.dtree.prune(X_TEST, self.labels_test)
        pruned_test_err = self.dtree.test_err(X_TEST, self.labels_test)
        pruned_train_err = self.dtree.train_err
        pruned_num_nodes = self.dtree.tree.num_nodes()

        assert pruned_test_err  < naive_test_err
        assert pruned_train_err >= naive_train_err
        assert pruned_num_nodes < naive_num_nodes


class TestDecisionTreeMissing:
    def setup(self):
        self.dtree = dt.DecisionTree()

    @raises(AttributeError)
    def test_classify(self):
        self.dtree.classify(X_TEST)

    @raises(AttributeError)
    def test_train_err_notree(self):
        self.dtree.train_err

    @raises(AttributeError)
    def test_test_err_notree(self):
        self.dtree.test_err(X_TEST, LABELS_TEST)

    @raises(AttributeError)
    def test_prune(self):
        self.dtree.prune(X_TEST, LABELS_TEST)


def use_bool_labels(cls):
    class newclass(cls):
        labels_train = LABELS_TRAIN_BOOL
        labels_test = LABELS_TEST_BOOL
    newclass.__name__ = cls.__name__ + 'Bool'
    return newclass

TestDecisionTreeFitBool = use_bool_labels(TestDecisionTreeFit)
TestEasyDecisionTreeErrorBool = use_bool_labels(TestEasyDecisionTreeError)
TestEasyDecisionTreeErrorBinBool = use_bool_labels(TestEasyDecisionTreeErrorBin)
TestEasyDecisionTreeErrorCatBool = use_bool_labels(TestEasyDecisionTreeErrorCat)
TestDecisionTreePruneBool = use_bool_labels(TestDecisionTreePrune)
