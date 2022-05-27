import numpy as np
from tqdm import tqdm

from scipy.stats import gamma, multivariate_normal
from .laplace import laplace_approximation_probit
from .approx_marginal_is import importance_sampler
from .kernels import chol_inverse

def mh_step(y, x, Kernel, th_old, marg_old, cov, N_imp = 100):
    """
    Performs one transition of the Pseudo-Marginal Metropolis Hastings algorithm.

    Parameters
    ----------
    y: N dimensional numpy vector
        responses
    
    x: N x D dimensional numpy array
        covariates
        
    Kernel: class
        kernel generating class

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
    # Constrain the old parameters
    th_old_constrained = Kernel.constrain_params(th_old)

    # Generate new unconstrained Kernel parameters from the proposal distribution
    th_new = np.random.multivariate_normal(th_old, cov, 1)[0]
    
    # Constrain the new parameters and update the kernel with them
    th_new_constrained = Kernel.constrain_params(th_new)
    new_kernel = Kernel(*th_new_constrained)
                             
    # Create the new kernel gram matrix, and make new Laplace approximation
    gram = new_kernel.make_gram(x, x)
    inverse_gram = chol_inverse(gram)
    laplace_mean, df_ll, laplace_cov, objective_history, converged = laplace_approximation_probit(y, inverse_gram)
    
    # Compute new marginal approximation
    marg_new = importance_sampler(y, x, laplace_mean, laplace_cov, N_imp)
    
    # Compute MH log ratio
    # Dimension of covariates for Gamma prior hyperameters in MH ratio
    d = x.shape[1]
    numer = marg_new +  Kernel.prior_log_pdf(th_new_constrained, d) + multivariate_normal.logpdf(th_old, mean = th_new)
    denom = marg_old +  Kernel.prior_log_pdf(th_old_constrained, d) + multivariate_normal.logpdf(th_new, mean = th_old)
    logratio = numer - denom
    
    # Check if we should move
    move = False
    u = np.random.uniform(0, 1, 1)
    if logratio > np.log(u):
        move = True
        return th_new, marg_new, move, inverse_gram
    else:
        return th_old, marg_old, move, inverse_gram
    
    
def mh(iters, y, x, Kernel, th_0, marg_0, cov, cov_scale=1, target_acc_rate=0.25, scale_iters=25, N_imp=64, verbose=True):
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
        
    Kernel: class
        kernel generating class

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
    inverse_gram_arr = np.zeros((iters, y.shape[0], y.shape[0]))
    
    acc_rate_hist = np.zeros(int(iters/scale_iters))
    cov_scale_hist = np.zeros(int(iters/scale_iters))
    cov_scale_hist[0] = cov_scale
    
    # Add initialisation
    th_arr[0, :] = th_0
    marg_arr[0] = marg_0
    
    for i in tqdm(range(iters), disable=not(verbose)):
        th_arr[i + 1, :], marg_arr[i + 1], move_arr[i], inverse_gram_arr[i, :, :] = mh_step(y, x, Kernel, th_arr[i, :], marg_arr[i], cov = cov_scale*cov, N_imp = N_imp)
        
        # Auto-tuning
        if (i != 0) & (i%scale_iters == 0):
            
            acc_rate = move_arr[(i-scale_iters):i].mean()
            acc_rate_hist[int(i/scale_iters)] = acc_rate
            
            # Make more drastic changes earlier in the iterations
            if acc_rate > target_acc_rate:
                cov_scale += 0.025 * (iters - i)/iters
            else:
                cov_scale -= 0.025 * (iters - i)/iters
            
            cov_scale_hist[int(i/scale_iters)] = cov_scale
            
    return th_arr, marg_arr, move_arr, acc_rate_hist, cov_scale_hist, inverse_gram_arr