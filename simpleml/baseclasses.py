from abc import ABCMeta, abstractmethod

class BinaryClassifier(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        ''' Fit the classifier using training data. '''
        return

    @abstractmethod
    def classify(self):
        ''' Classify new observations using the fitted classifier. '''
        return

    @classmethod
    def __subclasshook__(cls, C):
        if cls is not BinaryClassifier:
            return NotImplemented

        required_methods = ['fit', 'classify']
        for method in required_methods:
            if not any([method in B.__dict__ for B in C.__mro__]):
                return NotImplemented
        return True

