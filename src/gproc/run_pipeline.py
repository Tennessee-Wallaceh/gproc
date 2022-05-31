import numpy as np

from gproc.metrics import abstention_metrics

RHOS = np.linspace(0, 0.5, 1000)

def run_pipeline(fit_classifier, train_x, train_y, test_x, test_y):
    """
    Fits a classifier on train data and compute test statistics.
    
    Parameters
    ----------
    fit_classifier: function
        Fits a classifier which will provide a function for predicting unseen
        test points
    
    train_x: N x D numpy array
    
    train_y : num_observations x 1 numpy array
        array containing 1 or -1, the observations

    test_x: N x D numpy array
    
    test_y : num_observations x 1 numpy array
        array containing 1 or -1, the observations

    Returns
    ----------
    cap_metrics: numpy array
        A variety of abstention based metrics
    """
    predict, _ = fit_classifier(train_x, train_y)
    prob_y_positive = predict(test_x)
    predicted_y = np.ones_like(prob_y_positive)
    predicted_y[prob_y_positive < 0.5] = -1

    cap_metrics = abstention_metrics(prob_y_positive, predicted_y, test_y, RHOS)

    return cap_metrics