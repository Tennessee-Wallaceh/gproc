import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from gproc.ellss import slice_sample

def ess_step_cpp(f,K_chol,y):
    """
    Performs one transition of the Elliptic Slice Sampling algorithm.
    See http://proceedings.mlr.press/v9/murray10a/murray10a.pdf for details

    Parameters
    ----------
    f: N dimensional numpy vector
        previous sample from p(f | y, theta)

    K_chol: N x N numpy array
        lower triangular cholesky factor of kernel matrix of evaluated input locations

    y: N dimensional numpy vector
        y values from data

    Returns
    ----------
    f_dash:
        new sample from posterior :math:`p(f | y, theta)`
    """

    def L(f):
        return np.sum(norm.logcdf(y*f))

    #Auxilliary variate - specifies ellipse
    nu = (
          K_chol @
          np.random.multivariate_normal(np.zeros(K_chol.shape[0]), np.eye(K_chol.shape[0]))
          ).flatten()

    #Log-likelihood threshold for slice sampling
    u = np.random.uniform(0,1)
    log_y = L(f) + np.log(u)

    #Initial proposed angle
    angle = np.random.uniform(0,2*np.pi)

    #Define initial slice sampling bracket
    bracket = [angle - 2*np.pi , angle]


    #Generate a new sample f_dash using slice sampling. slice_sample updates f_dash in place
    f_dash = np.zeros(f.shape)
    slice_sample(f_dash, f, y, nu, bracket[0], bracket[1], log_y)
    return f_dash

def ess_samples_probit_cpp(K_chol, y, n_samples, burn_in):
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

    burn_and_samples = np.zeros((burn_in + n_samples , K_chol.shape[0]))

    for i in tqdm(range(1,burn_in + n_samples)):
        #if i%100 == 0:
            #print(f"~~~Sample {i} out of {burn_in + n_samples}~~~")
        burn_and_samples[i,:] = ess_step_cpp(burn_and_samples[i-1,:], K_chol, y)

    return burn_and_samples[burn_in:,:]
