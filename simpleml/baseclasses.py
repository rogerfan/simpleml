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
    def fit(self):
        ''' Fit the classifier using training data. '''
        return

    @abstractmethod
    def classify(self):
        ''' Classify new observations using the fitted classifier. '''
        return

    @abstractmethod
    def test_err(self):
        ''' Calculate the test error. '''
        return
