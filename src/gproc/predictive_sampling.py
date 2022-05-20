import numpy as np

from tqdm import tqdm

def blr_predictive(x_test, samples):
    """
    Sample from the Bayesian logisitic regression predictive distribution
    
    :param x_test, N x d matrix, matrix of test inputs
    :param samples, sample_size x d matrix, matrix of posterior samples
    
    :returns probs, N x sample_size matrix, matrix of predictive probabilities
    :returns y_preds, N x sample_size matrix, matrix of predictive samples
    :returns prob_ests, N vector, vector of predictive probability point estimates
    :returns y_ests, N vector, vector of prediction point estimates  
    """
    # Initialise arrays to hold predictive probabilities and predictive samples
    # for each test point and posterior sample
    y_preds = np.zeros((x_test.shape[0], samples.shape[1]))
    
    # For each test point, and posterior sample, compute the probability of label 1
    probs = 1/(1 + np.exp(-x_test.dot(samples)))
    y_preds[probs >= 0.5] = 1
    y_preds[probs < 0.5] = -1
    
    # Compute the point estimate of the probability that each test point is zero
    prob_ests = np.mean(probs, axis = 1)
    
    #Compute the point estimate of the label for each test point
    y_ests = np.mean(y_preds, axis = 1)
    y_ests[y_ests < 0] = -1
    y_ests[y_ests >= 0] = 1
    
    return probs, y_preds, prob_ests, y_ests
