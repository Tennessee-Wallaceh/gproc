import numpy as np
from scipy.stats import norm

from .kernels import squared_exponential

def sample_latent_f(gram_matrix):
    return np.random.multivariate_normal(0, gram_matrix)

def sample_at_x(x, kernel_fcn=squared_exponential, kernel_params={}, likelihood=norm.cdf):
    """
    :params x, N x d matrix of input locations
    """
    N = x.shape[0]
    gram_matrix = kernel_fcn(x, x, **kernel_params)
    f = np.random.multivariate_normal(np.zeros(N), gram_matrix)
    prob_y = likelihood(f)
    y = np.random.binomial(1, p=prob_y) # 1 sample for each N of elements of prob_y
    y[y == 0] = -1
    return y, prob_y, f