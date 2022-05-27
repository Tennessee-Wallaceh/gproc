"""Provides functions for performing Laplace approximations to GP classification posteriors."""

from scipy.stats import norm
import numpy as np

from .kernels import squared_exponential, chol_inverse

# The loglikelihood differentiated w.r.t each f
def ll_gradients(proposed_f, observed_y):
    pdf_cdf_ratio = norm.pdf(proposed_f) / norm.cdf(observed_y * proposed_f)
    df_ll = observed_y * pdf_cdf_ratio # N x 1
    df_2_ll = -np.square(pdf_cdf_ratio) - proposed_f * df_ll  # N x 1
    return df_ll, np.diag(df_2_ll)

def laplace_approximation_probit(observed_y, inverse_gram, max_iterations=100, tol=1e-5):
    """
    Computes the laplace approximation to the latent function implied by the model:

    .. math:: p(y_i | f_i) = \Phi(y_i * f_i)
    .. math:: p(f | K) = \mathcal{N}(0, K)

    Where :math:`\Phi` is the standard normal CDF and K is a gram matrix.

    We target the posterior:

    .. math:: p(f | y) \propto p(y | f)p(f | gram)

    with the Laplace approximation :math:`q(f) = \mathcal{N}(\mu, \Sigma)`.

    Parameters
    ----------
    y : num_observations x 1 numpy array
        array containing 1 or -1, the observations

    inverse_gram: num_observations x num_observations numpy array
        the precomputed inverse gram matrix corresponding to the observations

    Returns
    ----------
    :returns proposed_f: num_observations x 1 numpy array, the mean of the Laplace approximation  
    :returns df_ll: num_observations x 1 numpy array, the gradient of the log-likelihood wrt each :math:`f_i`
    """
    num_observations = observed_y.shape[0]

    _ll_gradients = lambda proposed_f: ll_gradients(proposed_f, observed_y)

    # The objective we are maximising log p(y | f) + log p(f | gram) + const
    def objective(proposed_f):
        log_likelihood = np.sum(norm.logcdf(observed_y * proposed_f))
        log_prior = -0.5 * proposed_f.dot(inverse_gram).dot(proposed_f)
        return log_likelihood + log_prior

    # Newton method update step (details in Rasmussen section 3.4)
    def update(proposed_f):
        df_ll, hessian = _ll_gradients(proposed_f)
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

def laplace_predict(new_x, x, gram, inverse_gram, laplace_mean, laplace_cov, df_ll,  kernel_fcn=squared_exponential, kernel_params = {}, pred_samples = 1000):
    """
    Make predictions using the laplace approximation to the posterior over the latent funcitons.

    Parameters
    ----------
    new_x: M x D numpy array
    
    x: N x D numpy array
    
    gram: N x N numpy array
    
    inverse_gram: N x N numpy array
    
    laplace_mean: N numpy vector

    laplace_cov: N x N numpy array
    
    df_ll: num_observations x 1 numpy array
        the gradient of the log-likelihood wrt each :math:`f_i`
        
    kernel_fcn: function: N x D numpy array, N x D numpy array, dictionary -> N x N numpy array
    
    pred_samples: float
        number of samples to generate from the predictive distribution corresponding to the laplace approximation
        to the latent function posterior
    
    Returns
    ----------
    predictive_mean: M numpy vector
        mean of predictive distribution
    
    predictive_cov: M x M numpy array
        covariance of predictive distribution
    
    predictive_y: M numpy vector
        averaged prediction
    """
    
    new_cross_gram = kernel_fcn(new_x, x, **kernel_params)
    new_gram = kernel_fcn(new_x, new_x, **kernel_params)

    inverse = np.linalg.inv(np.diag(1 / df_ll) + gram)

    predictive_mean = new_cross_gram.dot(inverse_gram).dot(laplace_mean)
    predictive_cov = new_gram - new_cross_gram.dot(inverse).dot(new_cross_gram.T)
    predictive_y = np.mean(norm.cdf(np.random.multivariate_normal(predictive_mean, predictive_cov, pred_samples)), axis=0)
    
    return predictive_y, predictive_mean, predictive_cov

def approximate_marginal_likelihood(x, y, kernel_fcn):
    """
    Computes the approximate marginal likelihood.

    Parameters
    ----------
    x: N x D numpy array
    
    y : num_observations x 1 numpy array
        array containing 1 or -1, the observations
    
    kernel_fcn: function: N x D numpy array, N x D numpy array, dictionary -> N x N numpy array

    Returns
    ----------
    approx_ml: float
        The approximate marginal likelihood for the provided kernel function
    """
    gram = kernel_fcn(x, x)
    inverse_gram = chol_inverse(gram)
    # Just want the first return value
    laplace_mean = laplace_approximation_probit(y, inverse_gram)[0]
    _, hessian = ll_gradients(laplace_mean, y)
    root_w = np.sqrt(-hessian)
    num_observations = x.shape[0]
    return (
        -0.5 * laplace_mean.dot(inverse_gram).dot(laplace_mean)
        -0.5 * np.log(np.linalg.det(np.eye(num_observations) + root_w.dot(gram).dot(root_w)))
        + np.sum(norm.logcdf(laplace_mean * y))
    )