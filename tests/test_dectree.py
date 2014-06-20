import numpy as np

import simpleml.dectree as dt


X_RAW = np.array([
    [-1.4273, -1.0824,  0.7958],
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

LABELS_RAW = np.array(
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
)


class TestEasyTree:
    def setup(self):
        self.x =  X_RAW
        self.labels = LABELS_RAW

        self.x[:,2] = np.abs(self.x[:,2])
        self.x[:,2] = self.x[:,2] * (-1+2*self.labels)

    def test_choose_split(self):
        from simpleml.metrics import gini
        best_split = dt._choose_split(self.x, self.labels, gini)

        x_sorted = np.sort(self.x[:,2])
        zero_ind = np.searchsorted(x_sorted, 0)

        assert best_split[0] == 2
        assert (best_split[1] == x_sorted[zero_ind] or
                best_split[1] == x_sorted[zero_ind-1])

    def test_create_tree(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        assert(tree.split[0] == 2)

        for i in range(2):
            assert tree.children[i].majority == i
            assert tree.children[1].split is None

    def test_tree_max_depth(self):
        tree = dt._create_decision_tree(self.x, self.labels, max_depth=0)
        assert tree.split is None

    def test_tree_descendents(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        assert len(tree.descendents()) == 3
        assert tree in tree.descendents()
        for i in range(2):
            assert tree.children[i] in tree.descendents()

    def test_data_at_note(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        assert np.all(
            self.x[self.x[:,2] < 0] == dt.data_at_node(tree, tree.children[0],
                                                       self.x)
        )
        assert np.all(
            self.x[self.x[:,2] > 0] == dt.data_at_node(tree, tree.children[1],
                                                       self.x)
        )

    def test_classify(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        for obs, label in zip(self.x, self.labels):
            assert tree.classify_obs(obs) == label

    def test_stumps(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        assert(len(tree.stumps()) == 1)
        assert(tree.stumps()[0] is tree)


class TestHarderTree:
    def setup(self):
        self.x = X_RAW
        self.labels = LABELS_RAW

    def test_tree_print(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        print(tree)

    def test_stumps(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        for stump in tree.stumps():
            for i in range(2):
                assert(stump.children[0].split is None)

    def test_min_obs_split(self):
        min_obs = 5
        tree = dt._create_decision_tree(self.x, self.labels,
                                        min_obs_split=min_obs)
        for desc in tree.descendents():
            if len(dt.data_at_node(tree, desc, self.x)) < min_obs:
                assert desc.split is None

    def test_classify(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        for obs, label in zip(self.x, self.labels):
            assert tree.classify_obs(obs) == label

