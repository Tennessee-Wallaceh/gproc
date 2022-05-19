import numpy as np
from scipy.spatial.distance import cdist

"""
As defined in https://www.cs.toronto.edu/~duvenaud/cookbook/
"""

def squared_exponential(x_1, x_2, lengthscale=0.5, variance=1.0):
    """
    Also known as RBF.
    :param x_1, N x d matrix
    :param x_2, M x d matrix

    :param lengthscale, float
    :param variance, float

    :returns K, N x M matrix, K_{ij} = k(x_i, x_j; lengthscale, variance)
    """
    sq_diffs = cdist(x_1, x_2, metric =' sqeuclidean')
    return variance * np.exp(-0.5 * sq_diffs / lengthscale)


