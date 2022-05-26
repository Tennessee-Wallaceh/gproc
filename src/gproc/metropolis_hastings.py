import numpy as np
from tqdm import tqdm

from scipy.stats import gamma, multivariate_normal
from .laplace import laplace_approximation_probit
from .kernels import squared_exponential, chol_inverse
from .approx_marginal_is import importance_sampler

def mh_step1(y, x, th_old, marg_old, cov, N_imp = 100):
    """
    Performs one transition of the Pseudo-Marginal Metropolis Hastings algorithm.

    Parameters
    ----------
    y: N dimensional numpy vector
        responses
    
    x: N x D dimensional numpy array
        covariates

    th_old: numpy vector
        contains old kernel parameters

    marg_old: float
        contains old approximation of the marginal
        
    cov: numpy array
        covariance matrix for use in the proposal distribution
        
    N_imp: float
        Number of importance samples to use in marginal approximation

    Returns
    ----------
    th_new: numpy vector
        new sample kernel parameters
    
    marg_new: float
        new marginal approximation
    
    move: boolean
        flag indicating whether or not we moved
    """
        
    # Draw kernel parameters from proposal distribution
    th_new = np.random.multivariate_normal(th_old, cov, 1).squeeze() # Remove axes of length 1

    # Reparameterise old and new kernel parameters
    l_old = np.exp(th_old[0])
    var_old = np.exp(th_old[1]) 
    
    l_new = np.exp(th_new[0])
    var_new = np.exp(th_new[1]) 
                             
    # Create new kernel matrix, and make new approximation
    gram = squared_exponential(x, x, lengthscale = l_new, variance = var_new)
    inverse_gram = chol_inverse(gram)
    laplace_mean, df_ll, laplace_cov, objective_history, converged = laplace_approximation_probit(y, inverse_gram)
    
    # Compute new marginal approximation
    marg_new = importance_sampler(y, x, laplace_mean, laplace_cov, N_imp)
    
    # Compute MH log ratio
    # Dimension for Gamma prior hyperameters in MH ratio
    d = x.shape[1]
    numer = marg_new +  gamma.logpdf(l_new, a = 1, scale = np.sqrt(d)) + gamma.logpdf(var_new, a = 1.2, scale = 1/0.2) + multivariate_normal.logpdf(th_old, mean = th_new)
    denom = marg_old +  gamma.logpdf(l_old, a = 1, scale = np.sqrt(d)) + gamma.logpdf(var_old, a = 1.2, scale = 1/0.2) + multivariate_normal.logpdf(th_new, mean = th_old)
    logratio = numer - denom
    
    # Check if we should move
    move = False
    u = np.random.uniform(0, 1, 1)
    if logratio > np.log(u):
        move = True
        return th_new, marg_new, move
    else:
        return th_old, marg_old, move
    
    
def mh1(iters, y, x, th_0, marg_0, cov, N_imp = 100, verbose = False):
    """
    Function that generates samples from the posterior distribution over
    kernel parameters.

    Parameters
    ----------
    iters: float
        number of iterations of the Metropolis Hastings algorithm
        
    y: N dimensional numpy vector
        responses
    
    x: N x D dimensional numpy array
        covariates

    th_0: numpy vector
        initial kernel parameters

    marg_0: float
        initial approximation of the marginal
        
    cov: numpy array
        covariance matrix for use in the proposal distribution
        
    N_imp: float
        number of importance samples to use in marginal approximation
    
    verbose: boolean
        flag to produce loading bar

    Returns
    ----------
    th_arr: numpy array
        contains the chains move history
    
    marg_arr: numpy array
        contains the history of marginal approximations
    
    acc_rate: float
        acceptance rate of moves
    """
    
    # Create array to hold samples and move history
    th_arr = np.zeros((iters + 1, th_0.shape[0]))
    marg_arr = np.zeros(iters + 1)
    move_arr = np.zeros(iters)
    
    # Add initialisation
    th_arr[0, :] = th_0
    marg_arr[0] = marg_0
    
    for i in tqdm(range(iters), disable=not(verbose)):
        th_arr[i + 1], marg_arr[i + 1], move_arr[i] = mh_step1(y, x, th_arr[i, :], marg_arr[i], cov = cov, N_imp = N_imp)
    
    acc_rate = move_arr.mean()
    return th_arr, marg_arr, acc_rate