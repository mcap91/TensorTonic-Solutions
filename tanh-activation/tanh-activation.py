import numpy as np
from numpy import exp

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=float)
    return ( exp(x) - exp(-x) ) / ( exp(x) + exp(-x) )
    pass