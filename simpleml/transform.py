'''
Data transformations.
'''
import numpy as np
from scipy.linalg import svd


def standardize(X):
    ''' Standardizes each column of the data to mean = 0, stdev = 1. '''
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


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
            num_comp = min(X.shape)
        else:
            self.num_comp = num_comp
        self.X = X
        self.Vt = None

    def fit(self):
        ''' Calculate the principle components. '''
        U, S, Vt = svd(self.X, full_matrices=False)
        self.Vt = Vt
        return self

    def transform(self, X=None, num_comp=None):
        ''' Apply dimensionality reduction.

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
        X, num_comp = self._check(X, num_comp)
        return np.dot(X, self.Vt[:self.num_comp].T)

    def project(self, X=None, num_comp=None):
        ''' Project onto the principle components.

        Parameters
        ----------
        X : ndarray [None]
            X matrix to project. Will use the trainig data if not provided.
        num_comp : ndarray [None]
            Number of principle components to use. If not provided, will use
            the number provided when initializing the PCA object.

        Returns
        -------
        X_new : ndarray
            Projected array of shape (num_samples, num_variables), the same
            shape as X.
        '''
        X, num_comp = self._check(X, num_comp)
        return np.dot(np.dot(X, self.Vt[:self.num_comp].T),
                      self.Vt[:self.num_comp])

    def _check(self, X, num_comp):
        if self.Vt is None:
            raise AttributeError('Model not yet fit. Use fit() first.')
        if X is None:
            X = self.X
        if num_comp is None:
            num_comp = self.num_comp

        return X, num_comp
