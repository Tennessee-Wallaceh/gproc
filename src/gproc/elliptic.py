import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def ess_step(f, K_chol, L):
    """
    Performs one transition of the Elliptic Slice Sampling algorithm.
    See http://proceedings.mlr.press/v9/murray10a/murray10a.pdf for details

    Parameters
    ----------
    f: N dimensional numpy vector
        previous sample from p(f | y, theta)

    K_chol: N x N numpy array
        lower triangular cholesky factor of kernel matrix of evaluated input locations

    L: function: N vector -> scalar, log-likelihood function wrapper that encloses the data y and just takes f as input.

    Returns
    ----------
    f_dash:
        new sample from posterior :math:`p(f | y, theta)`
    """

    #Auxilliary variate - specifies ellipse
    nu = (K_chol @ np.random.multivariate_normal(np.zeros(K_chol.shape[0]), np.eye(K_chol.shape[0]))).T

    #Log-likelihood threshold for slice sampling
    u = np.random.uniform(0,1)
    log_y = L(f) + np.log(u)

    #Initial proposed angle
    angle = np.random.uniform(0,2*np.pi)

    #Define initial slice sampling bracket
    bracket = [angle - 2*np.pi , angle]

    #Repeat until f that fits log-likelihood threshold is sampled
    valid_sample_found = False
    f_dash = None

    while not valid_sample_found:
        #New sample
        f_dash = f * np.cos(angle) + nu * np.sin(angle)

        #Check if valid
        if L(f_dash) > log_y:
            valid_sample_found = True

        #Else shrink bracket and choose new angle
        #Shrinking does nothing on first iteration but code is cleaner this way
        else:
            shrunk_side = int(angle >= 0) #0 or 1
            bracket[shrunk_side] = angle

            angle = np.random.uniform(*bracket)

    return f_dash


def ess_samples_probit(K_chol, y, n_samples, burn_in, f0=None, verbose=False):
    """
    Function that generates samples from the latent variables of the GP specified
    by a probit likelihood, the kernel matrix whose cholesky is given, and the y
    values given.

    Parameters
    ----------
    K_chol: N x N numpy array
        lower triangular cholesky factor of kernel matrix of evaluated input locations

    y: N dimensional numpy array
        Array of y values associated with the x values used to create the kernel matrix

    n_samples: Integer
        Number of samples that should be returned

    burn_in: Integer
        Length of the burn-in period, i.e. extra iterations that are not returned in the final output.

    Returns
    ----------
    samples: n_samples x N numpy array
        Samples of latent variables

    """
    if f0 is None:
        f0 = np.zeros(y.shape[0])
    
    #Closure of type N vector -> scalar , computing probit log-likelihood for
    #y given f, but as a function of f.
    def probit_L(f):
        return np.sum(norm.logcdf(y*f))

    burn_and_samples = np.zeros((burn_in + n_samples + 1 , K_chol.shape[0]))
    burn_and_samples[0, :] = f0

    for i in tqdm(range(1, burn_in + n_samples + 1), disable=not(verbose)):
        #if i%100 == 0:
            #print(f"~~~Sample {i} out of {burn_in + n_samples}~~~")
        burn_and_samples[i,:] = ess_step(burn_and_samples[i-1,:], K_chol, probit_L)

    return burn_and_samples[(burn_in + 1):,:]
