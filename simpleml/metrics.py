'''
Metric functions.
'''
import numpy as np


# Purity measures for binary variables.
def entropy(labels):
    prop = np.count_nonzero(labels) / len(labels)
    if prop == 0 or prop == 1:
        return 0.
    else:
        return -prop*np.log(prop) - (1-prop)*np.log(1-prop)

def gini(labels):
    prop = np.count_nonzero(labels) / len(labels)
    return prop*(1-prop)

def misclass(labels):
    prop = np.count_nonzero(labels) / len(labels)
    if np.round(prop) == 0:
        return prop
    else:
        return 1-prop

