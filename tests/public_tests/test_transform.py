import numpy as np
from nose.tools import raises

import simpleml.transform as tf


DATA1 = np.array(
    [[ 0.3, -2.6,  0.7, -1.3],
     [ 0.4, -1.8,  1.9, -0.4],
     [-0.2,  0.5, -0.4, -1.7],
     [-0.4,  3.5,  1.3, -1.3],
     [ 0.1, -0.6,  1.3, -1.3],
     [ 0.4, -2.8,  1.3,  0.8],
     [ 0.2, -1.5,  0.2, -1.5],
     [-0.3, -1.6,  1.5, -0.5]]
)

DATA2 = np.array(
    [[-0.4,  0.2, -0.6,  0.9],
     [ 1.6,  0.6, -1.1,  0.1],
     [-0.9, -0.7, -0.2,  0.9],
     [ 1.2,  0.5,  0.1,  0.3],
     [ 0.2, -0.6, -0.2, -0.9],
     [-0.3,  0.2,  1. , -0.5]]
)


class TestStandardize:
    def test_clean_data(self):
        data_std = tf.standardize(DATA1)

        assert np.allclose(np.mean(data_std, 0), np.zeros(DATA1.shape[1]))
        assert np.allclose(np.std(data_std, 0), np.ones(DATA1.shape[1]))

    def test_vec(self):
        data_std = tf.standardize(np.array([1, -1, 2, -2, 3, -3]))
        assert np.isclose(np.mean(data_std), 0)
        assert np.isclose(np.std(data_std), 1)

    @raises(ValueError)
    def test_missing(self):
        tf.standardize(np.array([1, -1, np.nan, -2, 3, -3]))

    @raises(ValueError)
    def test_inf(self):
        tf.standardize(np.array([1, -1, np.inf, -2, 3, -3]))

    @raises(ValueError)
    def test_const_mat(self):
        tf.standardize(np.ones((6, 4)))

    @raises(ValueError)
    def test_const_vec(self):
        tf.standardize(np.array([2, 2, 2]))


class TestPCA:
    def test_init_nonum(self):
        pca = tf.PCA().fit(DATA1)
        assert pca.num_comp == DATA1.shape[1]

    def test_init_withnum(self):
        pca = tf.PCA(num_comp=2)
        assert pca.num_comp == 2

    @raises(ValueError)
    def test_init_vec(self):
        tf.PCA().fit(np.array([1, 2, 3]))

    @raises(ValueError)
    def test_init_nan(self):
        tf.PCA().fit(np.array([[1, 2, 3], [1, np.nan, 3]]))

    @raises(ValueError)
    def test_init_inf(self):
        tf.PCA().fit(np.array([[1, 2, 3], [1, np.inf, 3]]))

    @raises(ValueError)
    def test_init_numtoobig(self):
        tf.PCA(num_comp=8).fit(DATA1)

    def test_project_allcomp_self(self):
        assert np.allclose(DATA1, tf.PCA().fit(DATA1).project(DATA1))

    def test_project_allcomp_other(self):
        assert np.allclose(DATA2, tf.PCA().fit(DATA1).project(DATA2))

    def test_project_allcomp_vec(self):
        vec = np.array([1, 2, 3, 4])
        assert np.allclose(vec, tf.PCA().fit(DATA1).project(vec))

    def test_project_shape1(self):
        data_proj = tf.PCA(num_comp=2).fit(DATA1).project(DATA1)
        assert data_proj.shape == DATA1.shape

    def test_project_shape2(self):
        data_proj = tf.PCA().fit(DATA1).project(DATA2, num_comp=2)
        assert data_proj.shape == DATA2.shape

    @raises(ValueError)
    def test_project_diff_shape(self):
        tf.PCA().fit(DATA1).project(DATA2[:,:2])

    @raises(ValueError)
    def test_project_numtoobig(self):
        tf.PCA().fit(DATA1).project(DATA1, num_comp=8)

    def test_transform_allcomp_self(self):
        data_trans = tf.PCA().fit(DATA1).transform(DATA1)
        assert data_trans.shape == DATA1.shape

    def test_transform_allcomp_other(self):
        data_trans = tf.PCA().fit(DATA1).transform(DATA2)
        assert data_trans.shape == DATA2.shape

    def test_transform_allcomp_vec(self):
        vec = np.array([1, 2, 3, 4])
        data_trans = tf.PCA().fit(DATA1).transform(vec)
        assert data_trans.shape == vec.shape

    def test_transform_shape1(self):
        data_trans = tf.PCA(num_comp=2).fit(DATA1).transform(DATA1)
        assert data_trans.shape == (DATA1.shape[0], 2)

    def test_transform_shape2(self):
        data_trans = tf.PCA().fit(DATA1).transform(DATA2, num_comp=2)
        assert data_trans.shape == (DATA2.shape[0], 2)

    @raises(ValueError)
    def test_transform_diff_shape(self):
        tf.PCA().fit(DATA1).transform(DATA2[:,:2])

    @raises(ValueError)
    def test_transform_numtoobig(self):
        tf.PCA().fit(DATA1).transform(DATA1, num_comp=8)
