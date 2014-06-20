import numpy as np

import simpleml.dectree as dt


class TestEasyTree:
    def setup(self):
        self.x = np.random.normal(size=(50, 5))
        self.labels = np.random.randint(0, 2, size=50)

        self.x[:,3] = np.abs(self.x[:,3])
        self.x[:,3] = self.x[:,3] * (-1+2*self.labels)

    def test_choose_split(self):
        from simpleml.metrics import gini
        best_split = dt._choose_split(self.x, self.labels, gini)

        x_sorted = np.sort(self.x[:,3])
        zero_ind = np.searchsorted(x_sorted, 0)

        assert best_split[0] == 3
        assert (best_split[1] == x_sorted[zero_ind] or
                best_split[1] == x_sorted[zero_ind-1])

    def test_create_tree(self):
        tree = dt._create_decision_tree(self.x, self.labels)
        assert(tree.split[0] == 3)

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
            self.x[self.x[:,3] < 0] == dt.data_at_node(tree, tree.children[0],
                                                       self.x)
        )
        assert np.all(
            self.x[self.x[:,3] > 0] == dt.data_at_node(tree, tree.children[1],
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
        self.x = np.random.normal(size=(50, 5))
        self.labels = np.random.randint(0, 2, size=50)

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

