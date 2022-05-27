import numpy as np
from .elliptic import ess_samples_probit
from .approx_marginal_is import importance_sampler
from .kernels import *
from .metropolis_hastings import mh
from tqdm import tqdm

def joint_sampler(iters, y, x, Kernel, th_0, marg_0, cov, cov_scale=1, target_acc_rate=0.25, scale_iters=25, N_imp=64, hyper_burn_in=100, ess_burn_in=25, verbose = True):
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
        
    Kernel: class
        kernel generating class

    th_0: numpy vector
        initial kernel parameters

    marg_0: float
        initial approximation of the marginal
        
    cov: numpy array
        covariance matrix for use in the proposal distribution
        
    target_acc_rate: float
        target acceptance rate
        
    scale_iters: float
        number of iterations per check for acceptance rate auto-tuning
        
    N_imp: float
        number of importance samples to use in marginal approximation
    
    hyper_burn_in: float
        number of burn in samples to use in MH hyperparameter sampling
        
    ess_burn_in: float
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
    
    print('Sampling hyperparameters')
    th_arr, marg_arr, move_arr, acc_rate_hist, cov_scale_hist, inverse_gram_arr = mh(iters, y, x, Kernel, th_0, marg_0, cov, cov_scale, target_acc_rate, scale_iters, N_imp, verbose)
    
    print('Burning in proposal latent function for ELL-SS algorithm')
    # Constrain the parameters of the chain state after burn-in
    th_constrained = Kernel.constrain_params(th_arr[hyper_burn_in, :])
        
    # Make kernel gram matrix with current constrained parameters
    kernel = Kernel(*th_constrained)
    gram = kernel.make_gram(x, x)
        
    # Sample once from the latent function, with significant burn-in
    K_chol = np.linalg.cholesky(gram+ 1e-05 * np.eye(gram.shape[0]))
    f_arr[hyper_burn_in - 1, :] = ess_samples_probit(K_chol, y, 1, 500, verbose = False)
     
    print('Sampling latent functions')
    for i in tqdm(range(hyper_burn_in, iters), disable=not(verbose)):
        # Constrain the parameters of the current chain state
        th_constrained = Kernel.constrain_params(th_arr[i, :])
        
        # Make kernel gram matrix with current constrained parameters
        kernel = Kernel(*th_constrained)
        gram = kernel.make_gram(x, x)
        
        # Sample once from the latent function
        K_chol = np.linalg.cholesky(gram+ 1e-05 * np.eye(gram.shape[0]))
        f_arr[i, :] = ess_samples_probit(K_chol, y, 1, ess_burn_in, f_arr[i - 1, :], verbose = False)
    
    acc_rate = move_arr.mean()
    return f_arr[hyper_burn_in:, :], th_arr, marg_arr, move_arr, acc_rate_hist, cov_scale_hist, inverse_gram_arr