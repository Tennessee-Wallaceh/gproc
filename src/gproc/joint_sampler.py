import numpy as np
from .elliptic import ess_samples_probit
from .approx_marginal_is import importance_sampler
from .kernels import *
from .metropolis_hastings import mh_step

def joint_sampler(iters, y, x, th_0, marg_0, cov, N_imp = 100, burn_in = 10, verbose = True):
    """
    Function that jointly samples from the posterior distribution over
    kernel parameters and the latent functions.

    Parameters
    ----------
    iters: float
        number of iterations of the joint sampling algorithm
        
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
    
    burn_in: float
        number of burn in samples to use in ELL-SS
        
    verbose: boolean
        flag to produce loading bar

    Returns
    ----------
    f_arr: numpy array
        contains the latent function samples
        
    th_arr: numpy array
        contains the kernel parmams chains move history
    
    marg_arr: numpy array
        contains the history of marginal approximations
    
    acc_rate: float
        acceptance rate of moves
    """
    
    # Create array to hold samples and move history
    f_arr = np.zeros((iters, y.shape[0]))
    th_arr = np.zeros((iters + 1, th_0.shape[0]))
    marg_arr = np.zeros(iters + 1)
    move_arr = np.zeros(iters)
    
    # Add initialisation
    th_arr[0, :] = th_0
    marg_arr[0] = marg_0
    
    for i in tqdm(range(iters), disable=not(verbose)):
        # Get latent function sample corresponding to current kernel params
        K = squared_exponential(x, x, lengthscale = th_arr[i, 0], variance = th_arr[i, 1])
        K_chol = np.linalg.cholesky(K + 1e-05 * np.eye(K.shape[0]))
        f_arr[i, :] = ess_samples_probit(K_chol, y, 1, burn_in, verbose = False)
        
        # Get new kernel params
        th_arr[i + 1, :], marg_arr[i + 1], move_arr[i] = mh_step(y, x, th_arr[i, :], marg_arr[i], cov = cov, N_imp = N_imp)
    
    acc_rate = move_arr.mean()
    return f_arr, th_arr, marg_arr, acc_rate