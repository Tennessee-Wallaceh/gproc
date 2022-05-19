from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
import numpy as np

JITTER = 1e-2 # Add so-called jitter for stability
def chol_inverse(X): 
    N = X.shape[0]
    chol = cho_factor(X + JITTER * np.eye(N), lower=True) 
    return cho_solve(chol, np.eye(N))

def laplace_approximation_probit(y, inverse_gram, max_iterations=100, tol=1e-5):
    """
    Computes the laplace approximation to the latent function implied by the model:

        p(y_i | f_i) = norm_cdf(y_i * f_i)
        p(f | gram) = normal(0, gram)

    We target the posterior:

        p(f | y) = Z * p(y | f) * p(f | gram)

    With an approximation q(f) = normal(mu, cov)

    :param y: N x 1 numpy array containing 1 or -1
    :param inverse_gram: N x N numpy array
    """
    N = y.shape[0]

    # The loglikelihood differentiated w.r.t each f
    def ll_gradients(f):
        pdf_cdf_ratio = norm.pdf(f) / norm.cdf(y * f)
        df_ll = y * pdf_cdf_ratio # N x 1
        df_2_ll = -np.square(pdf_cdf_ratio) - f * df_ll  # N x 1
        return df_ll, np.diag(df_2_ll)

    # The objective we are maximising log p(y | f) + log p(f | gram) + const
    def objective(f):
        ll = np.sum(norm.logcdf(y * f))
        l_prior = -0.5 * f.dot(inverse_gram).dot(f)
        return ll + l_prior

    # Newton method update step (details in Rasmussen section 3.4)
    def update(f):
        df_ll, hessian = ll_gradients(f)
        W = -hessian
        laplace_cov = chol_inverse(inverse_gram + W)
        return laplace_cov.dot(W.dot(f) + df_ll), df_ll, laplace_cov
    
    # Perform MAP of f using Newton method
    f = np.zeros(N) # Initialise at 0
    objective_history = np.zeros(max_iterations)
    objective_history[0] = objective(f)
    converged = False
    for i in range(1, max_iterations):
        f, df_ll, laplace_cov = update(f)
        objective_history[i] = objective(f)

        if np.abs(objective_history[i - 1] - objective_history[i]) < tol:
            converged = True
            break
    
    objective_history = objective_history[:i] # Slice down to the used iterations

    return f, df_ll, laplace_cov, objective_history, converged