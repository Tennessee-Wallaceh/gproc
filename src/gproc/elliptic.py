import numpy as np

def ess_step(f,K_chol,L):
    """
    Performs one transition of the Elliptic Slice Sampling algorithm.
    See http://proceedings.mlr.press/v9/murray10a/murray10a.pdf for details

    :param f, N vector, previous sample from p(f | y, theta)
    :param K_chol, N x N lower traignular matrix, lower cholesky factor of
                                                  kernel matrix of evaluated
                                                  input locations
    :param L, function: N vector -> scalar, log-likelihood function wrapper
                                            that encloses the data y and just
                                            takes f as input.

    :returns f_dash, new sample from posterior p(f | y, theta)
    """

    #Auxilliary variate - specifies ellipse
    nu = (
          K_chol @
          np.random.multivariate_normal(np.zeros(), np.eye(K_chol.shape[0]))
          ).T

    #Log-likelihood threshold for slice sampling
    u = np.random.uniform(0,1)
    log_y = L(f) + u

    #Initial proposed angle
    angle = np.random.uniform(0,2*np.pi)

    #Define initial slice sampling bracket
    bracket = (angle - 2*np.pi , angle)


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
