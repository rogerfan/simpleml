'''
Data transformations.
'''
import numpy as np
from scipy.linalg import svd


def standardize(X):
    '''
    Standardize each column of the data to mean = 0, stdev = 1.

    Note that this only handles clean, continuous data. Missing values should
    be handled before using this function, and constant variables should
    not be included. It is also probably not appropriate to use this function
    on binary or categorical variables, though it will mechanically standardize
    them as well.
    '''
    if not np.all(np.isfinite(X)):
        raise ValueError("Not all values are finite and non-missing.")

    means = np.mean(X, axis=0)
    stdevs = np.std(X, axis=0)

    if np.any(stdevs == 0):
        zero_ind = np.where(stdevs == 0.)[0]
        raise ValueError("Constant variable in columns {}.".format(zero_ind))

    return (X - means) / stdevs


class PCA:
    '''
    Principle Components Analysis.

    Parameters
    ----------
    X : ndarray
        Training data to estimate PCA on.
    num_comp : int [None]
        Number of principle components to use. If not provided, will default
        to min(num_samples, num_variables).

    Attributes
    ----------
    Vt : ndarray
        The transpose of the principle component weights.
    '''
    def __init__(self, X, num_comp=None):
        if num_comp is None:
            self.num_comp = min(X.shape)
        else:
            if num_comp > min(X.shape):
                raise ValueError("Number of components cannot be more than the"
                                 " number of variables or obs.")
            self.num_comp = num_comp
        self.X = X

        U, S, Vt = svd(X, full_matrices=False)
        self.Vt = Vt

    def transform(self, X=None, num_comp=None):
        '''
        Apply dimensionality reduction.

        Parameters
        ----------
        X : ndarray [None]
            X matrix to apply dimensionality reduction to. Will use the
            training data if not provided.
        num_comp : ndarray [None]
            Number of principle components to use. If not provided, will use
            the number provided when initializing the PCA object.

        Returns
        -------
        X_new : ndarray
            Transformed array of shape (num_samples, num_comp).
        '''
        X, num_comp = self._which_params_to_use(X, num_comp)
        return np.dot(X, self.Vt[:num_comp].T)

    def project(self, X=None, num_comp=None):
        '''
        Project onto the principle components.

        Parameters
        ----------
        X : ndarray [None]
            X matrix to project. Will use the training data if not provided.
        num_comp : ndarray [None]
            Number of principle components to use. If not provided, will use
            the number provided when initializing the PCA object.

        Returns
        -------
        X_new : ndarray
            Projected array of shape (num_samples, num_variables), the same
            shape as X.
        '''
        X, num_comp = self._which_params_to_use(X, num_comp)
        return np.dot(np.dot(X, self.Vt[:num_comp].T),
                      self.Vt[:num_comp])

    def _which_params_to_use(self, X, num_comp):
        if X is None:
            X = self.X
        if num_comp is None:
            num_comp = self.num_comp

        if num_comp > min(self.X.shape):
            raise ValueError("Number of components cannot be more than the "
                             "number of variables or obs in training data.")

        return X, num_comp
