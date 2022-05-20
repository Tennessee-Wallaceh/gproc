"""Provides functions for performing Laplace approximations to GP classification posteriors."""

from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import numpy as np

JITTER = 1e-5 # Add so-called jitter for stability of inversion

def chol_inverse(symmetric_x):
    """
    Computes the Cholesky decomposition x=LL^T, and uses this to compute the inverse of X.
    Only valid for symmetric x.

    :param x, dim_1 x dim_1 numpy array

    :returns x^{-1}, the inverse, such that x^{-1}X = I
    """
    dim_1 = symmetric_x.shape[0]
    chol = cho_factor(symmetric_x + JITTER * np.eye(dim_1), lower=True, check_finite=True)
    return cho_solve(chol, np.eye(dim_1))

def laplace_approximation_probit(observed_y, inverse_gram, max_iterations=100, tol=1e-5):
    """
    Computes the laplace approximation to the latent function implied by the model:

        p(y_i | f_i) = norm_cdf(y_i * f_i)
        p(f | gram) = normal(0, gram)

    We target the posterior:

        p(f | y) = Z * p(y | f) * p(f | gram)

    With an approximation q(f) = normal(mu, cov)

    :param y: num_observations x 1 numpy array containing 1 or -1
    :param inverse_gram: num_observations x num_observations numpy array
    """
    num_observations = observed_y.shape[0]

    # The loglikelihood differentiated w.r.t each f
    def ll_gradients(proposed_f):
        pdf_cdf_ratio = norm.pdf(proposed_f) / norm.cdf(observed_y * proposed_f)
        df_ll = observed_y * pdf_cdf_ratio # N x 1
        df_2_ll = -np.square(pdf_cdf_ratio) - proposed_f * df_ll  # N x 1
        return df_ll, np.diag(df_2_ll)

    # The objective we are maximising log p(y | f) + log p(f | gram) + const
    def objective(proposed_f):
        log_likelihood = np.sum(norm.logcdf(observed_y * proposed_f))
        log_prior = -0.5 * proposed_f.dot(inverse_gram).dot(proposed_f)
        return log_likelihood + log_prior

    # Newton method update step (details in Rasmussen section 3.4)
    def update(proposed_f):
        df_ll, hessian = ll_gradients(proposed_f)
        neg_hessian = -hessian
        laplace_cov = chol_inverse(inverse_gram + neg_hessian)
        return laplace_cov.dot(neg_hessian.dot(proposed_f) + df_ll), df_ll, laplace_cov

    # Perform MAP of f using Newton method
    proposed_f = np.zeros(num_observations) # Initialise at 0
    objective_history = np.zeros(max_iterations)
    objective_history[0] = objective(proposed_f)
    converged = False
    for i in range(1, max_iterations):
        proposed_f, df_ll, laplace_cov = update(proposed_f)
        objective_history[i] = objective(proposed_f)

        if np.abs(objective_history[i - 1] - objective_history[i]) < tol:
            converged = True
            break

    objective_history = objective_history[:i] # Slice down to the used iterations

    return proposed_f, df_ll, laplace_cov, objective_history, converged
