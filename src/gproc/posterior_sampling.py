import numpy as np
import pystan as stan
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def stanvert(x, y):
    """
    Converts data to pystan friendly format

    :params X, N x d matrix, matrix of features
    :params Y, N vector, vector of responses
    
    :stan_data, dict, stan data dictionary
    """
    
    stany = np.zeros(y.shape[0])
    stany[:] = y
    
    # Set the -1's to 0's for stan
    stany[stany == -1] = 0

    # Prepare the data for stan; x = covariates, y = response, d = dimensions, n = sample size
    stan_data = {'x': x, 'y': stany.astype(int), 'd': x.shape[1], 'n': x.shape[0]}
    
    return stan_data


def posterior_sampler(stan_model, stan_data, num_chains, num_samples, num_warmup):
    """
    Uses pystan to sample from a posterior distribution 
    
    :param stan_model, string, stan model code
    :param stan_data, dict, stan data dictionary
    :param num_chains, int, number of chains to sample
    :param num_samples, int, number of samples per chain
    :param num_warmup, int, number of warmup samples per chain
    
    :returns fit, stan.fit.Fit, fitted stan model containing posterior samples
    """
    posterior = stan.build(stan_model, data = stan_data)
    fit = posterior.sample(num_chains = num_chains, num_samples = num_samples, num_warmup = num_warmup)
    return fit



def inspect_samples(samples):
    """ 
    Plot the ACF, trace plots and marginal histograms of the posterior samples for each dimension
    
    :param samples, sample_size x d matrix, matrix of posterior samples  
    """
    d = samples.shape[1]
    fig, ax = plt.subplots(d, 3, figsize = (20, d*4))
    
    index = -1
    for i in range(d): 
        index += 1

        plot_acf(samples[:, i], lags = 40, ax = ax[index, 0])
        ax[index, 0].set_title(f'Autocorrelation (d = {i + 1})')
        ax[index, 1].plot(samples[:, i], linewidth = 0.1)
        ax[index, 1].set_title(f'Trace Plot (d = {i + 1})')
        ax[index, 2].hist(samples[:, i], density = True, bins = 30, ec = 'black')
        ax[index, 2].set_title(f'Marginal Histogram (d = {i + 1})')

    plt.show()