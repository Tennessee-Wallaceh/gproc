import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from .kernels import *
from .laplace import chol_inverse

def _chol_mvn_logpdf(f_samples, covariance):
    """
    Implements a multivariate normal log pdf using a cholesky decomposition.
    A D-dimensional multivariate normal.

    Parameters
    ----------
    f_sample: N x D
        0 mean examples, points at which to compute the log cdf

    covariance: D x D
    """
    inverse_cov, chol_cov = chol_inverse(covariance)
    log_cov_det = 2 * np.sum(np.log(np.diagonal(chol_cov)))
    return -0.5 * (
        log_cov_det +
        np.sum(f_samples.dot(inverse_cov) * f_samples, axis=1)
    )

def importance_sampler(y, x, q_mean, q_cov, N_imp, gram):
    """
    Implementation of eq 25 that approximates the marginal density p(y|th) on the log scale

    Parameters
    ----------
    iters: float
        number of iterations of the Metropolis Hastings algorithm
        
    y: N dimensional numpy vector
        responses
    
    x: N x D dimensional numpy array
        covariates

    q_mean: N vector
        mean of normal approximation to posterior over latents

    q_cov N x N numpy array
        covariance matrix of normal approximation to posterior over latents
        
    cov: numpy array
        covariance matrix for use in the proposal distribution
        
    N_imp: float
        number of importance samples to use in marginal approximation
    
    gram: N x N numpy array
        kernel gram matrix

    Returns
    ----------
    approximate marginal density, float
    
    """ 
    q_log_pdf = lambda f_samples: _chol_mvn_logpdf(f_samples - q_mean, q_cov)
    log_prior = lambda f_samples:  _chol_mvn_logpdf(f_samples, gram)
    log_likelihood = lambda f_samples: norm.logcdf(y * f_samples).sum(axis=1)

    f_samples = np.random.multivariate_normal(q_mean, q_cov, N_imp)

    return np.exp(
        log_likelihood(f_samples) +
        log_prior(f_samples) -
        q_log_pdf(f_samples)
    ).mean().log()