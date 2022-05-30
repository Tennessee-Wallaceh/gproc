import numpy as np
from scipy.stats import norm

def pm_predict(new_x, X, y, Kernel_class, f_arr, th_arr, inverse_gram_arr):

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
