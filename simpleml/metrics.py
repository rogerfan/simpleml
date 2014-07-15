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
    if len(prop) <= 1000:
        return np.where(prop <= 0.5, prop, 1-prop)
    else:  # Numexpr becomes faster around here (tested on i7-3770)
        return ne.evaluate('where(prop <= 0.5, prop, 1-prop)')
