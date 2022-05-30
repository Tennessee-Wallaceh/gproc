import numpy as np
from scipy.stats import norm

def pm_predict(new_x, X, y, Kernel_class, f_arr, th_arr, inverse_gram_arr):
    """
    Also known as RBF.

    Parameters
    ----------
    new_x: Size D numpy vector
        The new input point point to be predicted

    X: P x D numpy array
        The P training inputs of dimension D

    y: Size P numpy vector
        The P training targets

    Kernel_class: Class
        The class (not object) of the kernel the hyperparameters correspond to.

    f_arr: N x P numpy array
        The N samples of the latent function evaluated at the P training inputs.

    th_arr: N x C numpy array
        The N samples of the C hyperparameters

    inverse_gram_arr: N x P x P numpy array
        The N inverse Gram matrices we computed during the Metropolis-Hastings step,
        stored to trade memory for speedier predictions.

    Returns
    ----------
    pred: float
        The probability of the inputted point being in the positive class
    """
    
    N = f_arr.shape[0]

    mean_stars = np.empty(N)
    var_stars = np.empty(N)

    new_x_arr = new_x.reshape(1,new_x.shape[0])

    for i in range(N):
        constrained_params = Kernel_class.constrain_params(th_arr[i,:])
        kern = Kernel_class(*constrained_params)
        k_star = kern.make_gram(new_x_arr, X)
        k_star_star = kern.make_gram(new_x_arr, new_x_arr)

        m_star = k_star @ inverse_gram_arr[i,:,:] @ f_arr[i,:]
        s_star = k_star_star - k_star @ inverse_gram_arr[i,:] @ k_star.T

        mean_stars[i] = m_star.flatten()[0]
        var_stars[i] = s_star.flatten()[0]

    preds = norm.cdf(mean_stars / np.sqrt(1 + var_stars))

    return np.mean(preds)
