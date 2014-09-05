'''
Data transformations.
'''
import numpy as np
from scipy.linalg import svd

__all__ = ('standardize', 'PCA')


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
    num_comp : int, optional
        Number of principle components to use
        (default min(num_samples, num_variables)).
    '''
    def __init__(self, num_comp=None):
        self.num_comp = num_comp

    def fit(self, X):
        '''
        Fit the principle components model.

        Parameters
        ----------
        X : ndarray
            X matrix to extract principle components from.
        '''
        if self.num_comp is None:
            self.num_comp = min(X.shape)
        if self.num_comp > min(X.shape):
            raise ValueError("Number of components cannot be more than the"
                             " number of variables or observations.")

        U, S, Vt = svd(X, full_matrices=False)
        self.Vt = Vt
        self._var_num = X.shape[1]

        return self

    def transform(self, X, num_comp=None):
        '''
        Apply dimensionality reduction.

        Parameters
        ----------
        X : ndarray
            X matrix to apply dimensionality reduction to.
        num_comp : ndarray, optional
            Number of principle components to use (default self.num_comp).

        Returns
        -------
        X_new : ndarray
            Transformed array of shape (num_samples, num_comp).
        '''
        num_comp = self._verify_inputs(num_comp, X)
        return np.dot(X, self.Vt[:num_comp].T)

    def project(self, X, num_comp=None):
        '''
        Project onto the principle components.

        Parameters
        ----------
        X : ndarray
            X matrix to project. Will use the training data if not provided.
        num_comp : ndarray, optional
            Number of principle components to use (default self.num_comp).

        Returns
        -------
        X_new : ndarray
            Projected array of shape (num_samples, num_variables), the same
            shape as X.
        '''
        num_comp = self._verify_inputs(num_comp, X)
        return np.dot(np.dot(X, self.Vt[:num_comp].T),
                      self.Vt[:num_comp])

    def _verify_inputs(self, num_comp, X):
        if np.atleast_2d(X).shape[1] != self._var_num:
            raise ValueError("Number of variables for input data ({}) not the "
                             "same as training data ({})"
                             ".".format(X.shape[1], self._var_num))
        if num_comp is None:
            num_comp = self.num_comp
        if num_comp > self._var_num:
            raise ValueError("Number of components cannot be more than the "
                             "number of variables or obs in training data.")

        return num_comp
