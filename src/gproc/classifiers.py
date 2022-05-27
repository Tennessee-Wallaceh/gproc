from functools import reduce
import numpy as np

from gproc.laplace import laplace_approximation_probit, chol_inverse, laplace_predict, approximate_marginal_likelihood
from gproc.posterior_sampling import stanvert, blr_model, posterior_sampler
from gproc.predictive_sampling import blr_predictive

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
    """
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

    return predict

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
    """
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

    return predict

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

    """
    stan_data = stanvert(x_train, y_train)
    fit = posterior_sampler(blr_model, stan_data, num_chains, num_samples, num_burn_in)
    samples = fit['theta']

    def predict(x_new):
        probs, preds, prob_ests, y_ests = blr_predictive(x_new, samples)
        return prob_ests

    return predict