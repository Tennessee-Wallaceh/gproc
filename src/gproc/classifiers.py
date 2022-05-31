from functools import reduce
import numpy as np
from tqdm import tqdm

from gproc.laplace import laplace_approximation_probit, chol_inverse, laplace_predict, approximate_marginal_likelihood
from gproc.posterior_sampling import stanvert, blr_model, posterior_sampler
from gproc.predictive_sampling import blr_predictive
from gproc.joint_sampler import joint_sampler
from gproc.approx_marginal_is import importance_sampler
from gproc.pseudo_marginal_prediction import pm_predict

def simple_laplace(x_train, y_train, kernel_fcn):
    """
    Runs a laplace approximation with a fixed kernel and provides a function for predicting new x.

    Parameters
    ----------
    x_train: N x D numpy array
    
    y_train : num_observations x 1 numpy array
        array containing 1 or -1, the observations
    
    kernel_fcn: function: N x D numpy array, N x D numpy array, dictionary -> N x N numpy array

    Returns
    ----------
    predict: function
        Estimates probability of positive given test inputs

    fit_info: dict
        Dictionary of arbitrary info on the specific fit
    """
    fit_info = {}
    train_gram = kernel_fcn(x_train, x_train)
    inverse_train_gram = chol_inverse(train_gram)

    laplace_mean, df_ll, laplace_cov, _, converged = laplace_approximation_probit(y_train, inverse_train_gram)

    assert converged, 'Laplace approximation did not converge!'

    def predict(x_new):
        predictive_y, _, _ = laplace_predict(
            x_new,
            x_train,
            train_gram,
            inverse_train_gram,
            laplace_mean,
            laplace_cov,
            df_ll,
            kernel_fcn = kernel_fcn,
        )
        return predictive_y

    return predict, fit_info

def gridsearch_laplace(x_train, y_train, kernel_fcn, search_space):
    """
    Searches over potential hyper parameters and chooses the ones which maxmimise the approximate
    marginal log likelihood.
    Runs a laplace approximation with the best and provides a function for predicting new x.

    Parameters
    ----------
    x_train: N x D numpy array
    
    y_train : num_observations x 1 numpy array
        array containing 1 or -1, the observations
    
    kernel_fcn: function: N x D numpy array, N x D numpy array, dictionary -> N x N numpy array

    search_space: tuple
        Each element is an array corresponding to hyperparameter values to test.
        Order must correspond to order in kernel_fcn execution.

    Returns
    ----------
    predict: function
        Estimates probability of positive given test inputs

    fit_info: dict
        Dictionary of arbitrary info on the specific fit
    """
    fit_info = {}
    # Create array of size equal to all possible combinations of parameters
    marginal_lls = np.zeros(
        reduce(np.multiply, (len(param_set) for param_set in search_space)) 
    )

    # Create 1 x num_params array of possible combinations
    test_params = np.array(list(zip(
        param_set.flatten()
        for param_set in np.meshgrid(*search_space, indexing='ij')
    ))).squeeze().T

    for param_ix, test_param_set in enumerate(test_params):
        marginal_lls[param_ix] = approximate_marginal_likelihood(
            x_train,
            y_train,
            lambda x_1, x_2: kernel_fcn(x_1, x_2, *test_param_set)
        )

    # Get best Marginal LL and compute final gram
    best_ix = np.argmax(marginal_lls)
    best_params = test_params[best_ix]
    train_gram = kernel_fcn(x_train, x_train, *best_params)
    inverse_train_gram = chol_inverse(train_gram)
    laplace_mean, df_ll, laplace_cov, _, converged = laplace_approximation_probit(y_train, inverse_train_gram)
    assert converged, 'Laplace approximation did not converge!'

    def predict(x_new):
        predictive_y, _, _ = laplace_predict(
            x_new,
            x_train,
            train_gram,
            inverse_train_gram,
            laplace_mean,
            laplace_cov,
            df_ll,
            kernel_fcn=lambda x_1, x_2: kernel_fcn(x_1, x_2, *best_params),
        )
        return predictive_y

    return predict, fit_info

def bayesian_logreg(x_train, y_train, num_chains=1, num_samples=2500, num_burn_in=500):
    """
    Estimates parameters for a Bayesian logistic regression model using Stan.
    
    Parameters
    ----------
    x_train: N x D numpy array
    
    y_train : num_observations x 1 numpy array
        array containing 1 or -1, the observations

    num_chains: int
        The number of independent chains to sample.

    num_samples: int
        The number of samples in each chain

    num_burn_in: int
        The number of samples to throw away at the start of the chain

    Returns
    ----------
    predict: function
        Estimates probability of positive given test inputs

    fit_info: dict
        Dictionary of arbitrary info on the specific fit
    """
    fit_info = {}
    stan_data = stanvert(x_train, y_train)
    fit = posterior_sampler(blr_model, stan_data, num_chains, num_samples, num_burn_in)
    samples = fit['theta']

    def predict(x_new):
        probs, preds, prob_ests, y_ests = blr_predictive(x_new, samples)
        return prob_ests

    return predict, fit_info

def full_bayesian_pseudo_marginal(x_train, y_train, Kernel):
    """
    Performs full Bayesian inference in pseudo-marginal framework

    Parameters
    ----------
    x_train: N x D numpy array
    
    y_train : num_observations x 1 numpy array
        array containing 1 or -1, the observations
    
    Kernel: Class
        Class inheriting from gproc.kernel.BaseKernel
        Defines family of kernels

    Returns
    ----------
    predict: function
        Estimates probability of positive given test inputs

    fit_info: dict
        Dictionary of arbitrary info on the specific fit
    """

    fit_info = {}

    # Get initial estimate
    th_0 = np.zeros(Kernel.param_dim) # The initial unconstrained kernel parameters
    th_0_constrained = Kernel.constrain_params(th_0)
    initial_kernel = Kernel(*th_0_constrained)
    gram = initial_kernel.make_gram(x_train, x_train)
    inverse_gram = chol_inverse(gram)
    laplace_mean, df_ll, laplace_cov, _, converged = laplace_approximation_probit(y_train, inverse_gram)
    
    assert converged, f'Initial Laplace approximation did not converge!'

    marg_0 = importance_sampler(y_train, x_train, laplace_mean, laplace_cov, 100)

    # Run joint sample
    f_arr, th_arr, marg_arr, move_arr, acc_rate_hist, cov_scale_hist, inverse_gram_arr = joint_sampler(
        5000, y_train, x_train, Kernel, th_0, marg_0, cov=np.eye(th_0.shape[0]), 
        N_imp=64, hyper_burn_in=1000, ess_burn_in=1, verbose=True
    )

    def predict(x_new):
        N_new = x_new.shape[0]
        predictive_y = np.empty(N_new)
        for i in tqdm(range(N_new)):
            predictive_y[i] = pm_predict(x_new[i,:], x_train, y_train, Kernel, f_arr, th_arr, inverse_gram_arr)

        return predictive_y

    return predict, fit_info
