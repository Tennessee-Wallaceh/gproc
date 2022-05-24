from scipy.linalg import cho_factor, cho_solve
import numpy as np

from .kernels import *
from .laplace import chol_inverse 

JITTER = 1e-5 # Add so-called jitter for stability of inversion

def importance_sampler(y, x, q_mean, q_cov, N_imp, kernel_fcn=squared_exponential, kernel_params={}):
    """
    Implementation of eq 25 that approximates the marginal density p(y|th) on the log scale
    
    :param y: num_observations x 1 numpy array containing 1 or -1
    :param x, N x d matrix of input locations
    :param q_mean: num_observations x 1 numpy array, mean of normal approximation to posterior over latents
    :param q_cov: num_observations x num_observations numpy array, covariance matrix of normal approximation to posterior over latents
    :param N_imp: float, number of inmportance samples to use
    
    :returns approximate marginal density, float
    """
    # Get gram matrix
    gram = kernel_fcn(x, x, **kernel_params)
    
    # Inverse of gram matrix 
    inverse_gram = chol_inverse(gram)
    
    # Work out log-determinant of gram matrix
    chol_gram = cho_factor(gram + JITTER * np.eye(gram.shape[0]), lower=True, check_finite=True)[0]
    log_gram_det = 2 * np.sum(np.log(np.diagonal(chol_gram)))
    
    # Get the log determinant and inverse of the approximate covariance matrix
    chol_q_cov = cho_factor(q_cov + JITTER * np.eye(q_cov.shape[0]), lower=True, check_finite=True)[0]
    log_q_det = 2 * np.sum(np.log(np.diagonal(chol_q_cov)))
    inverse_q_cov = chol_inverse(q_cov)
    
    # Define log joint function
    def log_joint(y, f, inverse_gram, log_gram_det):
        log_likelihood = np.sum(norm.logcdf(y * f))
        log_prior = -0.5 * ( y.shape[0] * np.log(2 * np.pi) + log_gram_det + f.dot(inverse_gram).dot(f) )
        return log_likelihood + log_prior

    # Define log density of Normal approximation q
    def q_log_pdf(f, inverse_q_cov, log_q_det):
        return -0.5 * ( f.shape[0] * np.log(2 * np.pi) + log_q_det + f.dot(inverse_q_cov).dot(f) )

    # Sample latent functions from the normal approximation to full posterior
    f_samples = np.random.multivariate_normal(q_mean, q_cov, N_imp)
    
    # Importance sum
    marg_approx = 0
    for i in range(N_imp):
        marg_approx += log_joint(y, f_samples[i, :], inverse_gram, log_gram_det) - q_log_pdf(f_samples[i, :], inverse_q_cov, log_q_det)
    
    return marg_approx/N_imp