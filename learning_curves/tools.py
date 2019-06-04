import dill
import numpy as np
import matplotlib.pyplot as plt
from .learning_curves import learning_curve

def load(path="./lc_data.pkl"):
    """ Load a :class:`learning_curves.LearningCurve` object from disk. """
    with open(path, 'rb') as f:
        return dill.load(f)


def is_strictly_increasing(L):
    """ Returns True if the list contains strictly increasing values. 
    
        Examples: 
            is_strictly_increasing([0,1,2,3,4,5]) > True
            is_strictly_increasing([0,1,2,2,4,5]) > False
            is_strictly_increasing([0,1,2,1,4,5]) > False
    """
    for x, y in zip(L, L[1:]):
        if x > y: return False
    return True


def get_scale(val, floor=True):
    """ Returns the scale of a value. 

        Args:
            floor (bool): if True, apply np.floor to the result 

        Examples: 
            get_scale(1.5e-15) > -15 
            get_scale(1.5e-15, False) > -14.823908740944319
    """
    val = np.log10(np.abs(val))
    return np.floor(val) if floor else val
     

def get_unique_list(predictors):
    """ Return a list of unique predictors. Two Predictors are equal if they have the same name."""
    results = []
    for P in predictors:
        #if not P.name in [p.name for p in results] : results.append(P)
        if P not in results : results.append(P)
    return results


def update_params(params, strategies):
    """ Update the values of params based on the values in strategies. 
    
        Example: update_params(params=dict(val1=1, val2=10), strategies=dict(val1=0.1, val2=-1)
            > {'val1': 1.1, 'val2': 9}
    """
    for key, value in strategies.items():
        if key in params: 
            params[key] += value
    return params


def get_absolute_value(validation, len_vector):
    """ Get the absolute value of the length of a vector. """
    assert validation >= 0, "validation parameter must be between 0 and 1, or positive integer."
    if isinstance(validation, float): return int(validation * len_vector)
    elif np.isscalar(validation) and validation > 0: return validation
    else: raise ValueError("validation parameter must be between 0 and 1, or positive integer.")


def mean_bias_error(y_trues, y_preds):
    """ Computes the Mean Bias Error of two vectors. """
    return np.mean(y_trues - y_preds)
