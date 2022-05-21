"""
Functions for generating example classification problems.
"""
import numpy as np
from scipy.stats import norm

from .kernels import squared_exponential

def sample_at_x(x, kernel_fcn=squared_exponential, kernel_params={}, likelihood=norm.cdf):
    """
    Samples a class label at each provided input point according to the GP classification model.
    Allows arbitrary kernel function and likelihood.

    Parameters
    ----------
    x: N x D numpy array
        Array of the input locations at which to draw samples.
    
    kernel_fcn: fcn: :math:`\mathbb{R}^{N × D} × \mathbb{R}^{M × D} → \mathbb{R}^{N × M}`
        Kernel function which produces gram matrix given two sets of inputs

    kernel_params: dict
        kwargs to pass on to the kernel function

    likelihood: fcn: :math:`\mathbb{R}^{N} → (0, 1)^{N}`
        The likelihood of a positive samples given the latent function :math:`p(y=1 | f)`

    Returns
    ----------
    y: N x 1 numpy integer array
        The sampled class labels, either +1 or -1

    prob_y: N x 1 numpy array
        The probability of a positive sample at each point
    
    f: N x 1 numpy array
        The latent function value at each input point.
    """
    N = x.shape[0]
    gram_matrix = kernel_fcn(x, x, **kernel_params)
    f = np.random.multivariate_normal(np.zeros(N), gram_matrix)
    prob_y = likelihood(f)
    y = np.random.binomial(1, p=prob_y) # 1 sample for each N of elements of prob_y
    y[y == 0] = -1
    return y, prob_y, f