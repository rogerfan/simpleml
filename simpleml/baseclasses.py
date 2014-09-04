'''
Abstract base classes.
'''
from abc import ABCMeta, abstractmethod


class AbstractBase(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, C):
        for method in cls._req_meth:
            if not any([method in B.__dict__ for B in C.__mro__]):
                return NotImplemented
        return True


class BinaryClassifier(AbstractBase):
    _req_meth = ('fit', 'classify', 'test_err')

    @abstractmethod
    def fit(self, X, Y):
        ''' Fit the classifier using training data. '''
        return

    @abstractmethod
    def classify(self, X):
        ''' Classify new observations using the fitted classifier. '''
        return

    @abstractmethod
    def test_err(self, X, Y):
        ''' Calculate the test error. '''
        return

class Function(AbstractBase):
    _req_meth = ('f', 'd')

    @abstractmethod
    def f(self, x):
        ''' Calculate function values. '''
        return

    @abstractmethod
    def d(self, x):
        ''' Calculate derivative values. '''
        return
