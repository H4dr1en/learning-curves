import dill
import numpy as np


def load(path="./lc_data.pkl"):
    """ Load a LearningCurve object from disk. """
    with open(path, 'rb') as f:
        return dill.load(f)


def is_strictly_increasing(L):
    """ Returns True if the list contains strictly increasing values. """
    for x, y in zip(L, L[1:]):
        if x > y: return False
    return True


def get_scale(val):
    """ Returns the scale of a value. Eg: get_scale(1e-15) = -15 """
    return np.floor(np.log10(np.abs(val)))
    

def get_unique_list(predictors):
    """ Return a list of unique predictors. """
    results = []
    for P in predictors:
        if not P.name in [p.name for p in results] : results.append(P)
    return results


    