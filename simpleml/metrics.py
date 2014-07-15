'''
Metric functions.
'''
import numpy as np
import numexpr as ne

# Purity measures for binary variables.
def entropy(prop):
    return ne.evaluate('where((prop != 0) & (prop != 1),'
                       '-prop*log(prop) - (1-prop)*log(1-prop), 0)')

def gini(prop):
    return prop*(1-prop)

def misclass(prop):
    if hasattr(prop, '__len__') and len(prop) > 2000:
        return ne.evaluate('where(prop <= 0.5, prop, 1-prop)')
    else:  # Numexpr is slower for small arrays (tested on i7-3770)
        return np.where(prop <= 0.5, prop, 1-prop)
