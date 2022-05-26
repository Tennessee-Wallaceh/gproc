"""
A number of python kernels, as defined in https://www.cs.toronto.edu/~duvenaud/cookbook/
"""

import numpy as np
from scipy.spatial.distance import cdist
from functools import reduce

def squared_exponential(x_1, x_2, lengthscale=0.5, variance=1.0):
    """
    Also known as RBF.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    sq_diffs = cdist(x_1, x_2, metric ='sqeuclidean')
    return variance * np.exp(-0.5 * sq_diffs / lengthscale)

def rational_quadratic(x_1, x_2, lengthscale=0.5, variance=1.0, weighting=1.0):
    """
    Rational Quadratic Kernel, equivalent to adding together many Squared Exponential kernels with different 
    lengthscales. Weight parameter determine relative weighting of large and small scale variations. When
    the weighting goes to infinity, RQ = SE.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    weighting: float
        Relative weighting of large and small scale variations

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    sq_diffs = cdist(x_1, x_2, metric = 'sqeuclidean')
    return variance * ( (1 + sq_diffs / 2*lengthscale * weighting) ** (-weighting) )

def periodic(x_1, x_2, lengthscale=0.5, variance=1.0, period=1.0):
    """
    The periodic kernel allows one to model functions which repeat themselves exactly.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    period: float
        Distance between repititions

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    diffs = cdist(x_1, x_2, metric = 'euclidean')
    return variance * np.exp(-0.5 * np.sin(np.pi * diffs / period) / lengthscale)

def locally_periodic(x_1, x_2, lengthscale=0.5, variance=1.0, period=1.0):
    """
    A squared exponential kernel multiplied by a periodic kernel. Allows one to model periodic functions
    which can vary slowly over time.

    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array

    lengthscale: float
        Lower lengthscale gives "wigglier" functions
    
    variance: float
        The average distance from the mean

    period: float
        Distance between repititions

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    
    return np.multiply(
        squared_exponential(x_1, x_2, lengthscale, variance), 
        periodic(x_1, x_2, lengthscale, variance, period)
    )

def linear(x_1, x_2, constant_variance=0.5, variance=1.0, offset=1.0):
    """
    A linear kernel is a non-stationary kernel, which when used with a GP, is equivalent to
    Bayesian linear regression.
    
    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array
    
    constant_variance: float
        Controls height of function at 0

    variance: float
        The average distance from the mean

    offset: float
        x coordinate of location all functions go through

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j)`
    """
    return constant_variance + variance * np.dot(x_1 - offset, x_2.T - offset)

def add(x_1, x_2, kernels):
    """
    Add together an arbitrary number of kernels.
    
    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array
    
    kernels: iterable((kernel_fnc, kernel_parameters), ...)
        A number of kernel function and parameter tuples.

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j) = \sum_{l=1}^{L}k_l(x_i, x_j)`,
        for L kernels.
    """
    return sum([
        kernel(x_1, x_2, **kernel_kwargs)
        for kernel, kernel_kwargs in kernels
    ])

def multiply(x_1, x_2, kernels):
    """
    Multiply together an arbitrary number of kernels.
    
    Parameters
    ----------
    x_1: N x D numpy array
    
    x_2: M x D numpy array
    
    kernels: iterable((kernel_fnc, kernel_parameters), ...)
        A number of kernel function and parameter tuples.

    Returns
    ----------
    K: N x M matrix,
        The corresponding kernel matrix :math:`K_{ij} = k(x_i, x_j) = \prod_{l=1}^{L}k_l(x_i, x_j)`,
        for L kernels.
    """
    
    grams = [
        kernel(x_1, x_2, **kernel_kwargs) 
        for kernel, kernel_kwargs in kernels
    ]
    
    return reduce(np.multiply, grams)
