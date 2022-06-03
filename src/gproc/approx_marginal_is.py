import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from .kernels import *
from .laplace import chol_inverse 

JITTER = 1e-5 # Add so-called jitter for stability of inversion

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
    # Inverse of gram matrix and cholesky decomp
    inverse_gram, chol_gram = chol_inverse(gram)
    
    # Work out log-determinant of gram matrix
    log_gram_det = 2 * np.sum(np.log(np.diagonal(chol_gram)))

    # Get the log determinant and inverse of the approximate covariance matrix
    chol_q_cov = cho_factor(q_cov + JITTER * np.eye(q_cov.shape[0]), lower=True, check_finite=True)[0]
    log_q_det = 2 * np.sum(np.log(np.diagonal(chol_q_cov)))
    inverse_q_cov, _ = chol_inverse(q_cov)
    
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