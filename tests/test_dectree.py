import numpy as np

import simpleml.dectree as dt


X_RAW = np.array([
    [-1.4273, -1.0824,  0.7958, -0.4138,  1.7358],
    [ 0.7866, -0.3452, -0.5211,  0.5205,  0.0808],
    [ 1.1191,  0.0517, -1.1401,  1.2768, -1.2073],
    [ 1.6514,  0.2658,  0.4943, -0.778 , -0.7284],
    [ 0.3742, -0.9247,  0.1196, -0.7706,  0.3882],
    [-0.5938, -0.9151, -0.2263, -1.1365,  0.4649],
    [ 1.0806, -0.5539, -1.9748, -1.2799, -1.5339],
    [-0.2261,  0.228 ,  0.5221,  0.2902, -0.5123],
    [ 1.0467,  0.9656,  0.4729,  0.2745, -0.0471],
    [-1.327 , -0.7972, -1.4782,  0.0522,  0.314 ],
    [ 0.6219, -0.0472, -1.178 ,  0.4999, -0.978 ],
    [ 0.4702,  0.16  , -0.111 , -0.1407, -0.2204],
    [-0.7008, -0.586 , -1.2022, -1.7318,  1.3425],
    [-0.0925,  0.5194,  1.7272,  0.1604, -1.1906],
    [ 1.6551, -0.3479, -2.0299,  1.5493,  1.3885],
    [-1.0322, -0.6711, -1.476 , -0.4675,  0.0274],
    [ 0.6841, -1.2086, -1.6619, -0.0431,  1.0827],
    [ 0.3332, -1.8284, -0.291 , -1.7727,  0.0672],
    [-1.8365,  1.2062, -0.888 ,  0.593 ,  1.3286],
    [-0.2278, -1.2292,  1.5369,  1.4117, -0.1072],
    [ 1.3067, -1.2598,  0.1791, -0.3083, -1.0505],
    [ 0.9992, -0.3767, -0.2041,  1.4425,  0.3425],
    [ 1.0983,  1.6152, -0.2454, -0.6029, -0.4128],
    [ 1.8958, -0.4772,  0.5111, -0.6376, -0.0895],
    [-2.0339,  0.2129, -1.5437, -0.3104, -0.3912],
    [-0.9744,  1.0371, -1.1812,  0.7166,  1.0911],
    [-2.3093,  0.3592, -1.3685,  0.166 ,  1.8034],
    [-0.9211,  0.1753, -1.0368, -2.5384,  0.5449],
    [-0.4949, -1.1877,  0.6837, -0.3339,  2.137 ],
    [-0.592 ,  1.154 ,  0.2365, -0.3842,  1.6334],
    [-0.9842, -0.0577,  1.8842,  0.9521, -0.7841],
    [-0.5876,  0.5185, -1.4788, -0.6272,  1.5313],
    [ 1.0457,  1.9564, -1.8296,  1.4111,  0.5124],
    [ 1.7437, -1.2161, -0.2377,  0.7257, -2.8028],
    [-0.1008,  0.1122, -0.184 , -0.467 ,  1.2184],
    [ 0.7119, -0.0854,  0.1952,  1.1288, -2.1937],
    [ 0.0604, -0.3383, -0.9218,  0.5755,  0.9961],
    [-1.2439,  0.5794, -0.9363,  2.3625, -0.845 ],
    [ 1.674 ,  1.4144,  1.0524,  0.6743, -0.0748],
    [-0.0194, -1.3548, -0.1377,  0.0553, -0.474 ],
    [-0.4417, -0.9871, -0.8109, -0.5775,  1.4598],
    [ 0.8786,  0.6812, -1.2233, -1.7027, -1.4576],
    [ 1.4511, -0.068 ,  0.2631, -0.0997, -0.4104],
    [-0.0604,  0.7227, -0.7551,  0.5814, -1.2637],
    [-0.1787,  0.6172, -0.074 , -1.8587, -0.5649],
    [ 1.8281,  0.813 , -1.0849,  2.3434, -0.7402],
    [-0.1834, -1.0187, -0.0787,  1.6151, -0.1673],
    [ 0.9016, -1.5295, -0.7564,  1.0032, -0.3703],
    [ 1.5219, -1.1504, -1.0542,  1.6848,  2.3348],
    [ 1.0251,  0.4972, -0.5228, -0.4864, -0.1213]
])

LABELS_RAW = np.array([
    1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
    0, 0, 1, 0
])


class TestEasyTree:
    def setup(self):
        self.x =  X_RAW
        self.labels = LABELS_RAW

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

