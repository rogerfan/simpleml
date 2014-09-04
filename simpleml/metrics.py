'''
Metric functions.
'''
import numpy as np
import numexpr as ne

__all__ = ('entropy', 'gini', 'misclass', 'tanh', 'logistic')


# Purity measures for binary variables.
def entropy(prop):
    return ne.evaluate('where((prop != 0) & (prop != 1),'
                       '-prop*log(prop) - (1-prop)*log(1-prop), 0)')

def gini(prop):
    return prop*(1-prop)

def misclass(prop):
    if hasattr(prop, '__len__') and len(prop) > 5000:
        return ne.evaluate('where(prop <= 0.5, prop, 1-prop)')
    else:  # Numexpr is slower for small arrays (tested on i7-3770)
        return np.where(prop <= 0.5, prop, 1-prop)

# Sigmoid functions with derivatives
class tanh:
    def f(self, x):
        return np.tanh(x)
    def d(self, x):
        return 1. - x**2
tanh = tanh()

class logistic:
    def f(self, x):
        return 1. / (1. + np.exp(-x))
    def d(self, x):
        f = self.f(x)
        return f*(1.-f)
logistic = logistic()
