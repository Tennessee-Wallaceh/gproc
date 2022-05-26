import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

class ThresholdError(Exception):
    pass

def predictive_metrics(prob_ests, y_ests, y_test, rho):
    """
    Computes the capacity accuracy, capacity F score and capacity AUC scores
    
    :param prob_ests, N vector, vector of predictive probability point estimates
    :param y_ests, N vector, vector of prediction point estimates  
    :param y_tests, N vector, vector of test responses
    :param rho, float[0, 1], prediction certainity threshold value
    
    :returns abstention_degree, float[0. 1], proportion of test data retained after thresholding
    :returns cap_acc_score, float[0, 1], capacity classification accuracy
    :returns cap_F_score, float[0, 1], capacity F score
    :returns cap_AUC_score, float[0, 1], capacity AUC score
    """
    # Number of test points
    N = y_test.shape[0]
    
    # Find the indices that pass the certainty threshold
    thresh_inds = np.argwhere((prob_ests < 0.5 - rho) | (prob_ests > 0.5 + rho))
    
    # Work out degree of abstention
    abstention_degree = thresh_inds.shape[0]/N

    if abstention_degree == 0:
        raise ThresholdError("Threshold value too large, no data passes")
    
    # Get thresholded predicted and test responses, and probabilities
    y_test_threshed = y_test[thresh_inds]
    y_ests_threshed = y_ests[thresh_inds]
    prob_ests_thresed = prob_ests[thresh_inds]
    
    if (np.unique(y_test_threshed).shape[0] == 1) | (np.unique(y_ests_threshed).shape[0] == 1):
        raise ThresholdError("Threshold value too large, only one class")
        
    # Work out the capacity classifcication accuracy
    cap_acc_score = np.mean(y_test_threshed == y_ests_threshed)
    
    # Work out the capacity  F score 
    cap_f_score = f1_score(y_test_threshed, y_ests_threshed)
    
    # Work out the capacity  AUC score
    cap_auc_score = roc_auc_score(y_test_threshed, prob_ests_thresed)
    
    return abstention_degree, cap_acc_score, cap_f_score, cap_auc_score

def abstention_metrics(prob_ests, y_ests, y_test, rhos):
    """
    Computes the capacity accuracy, capacity F score and capacity AUC scores
    
    :param prob_ests, N vector, vector of predictive probability point estimates
    :param y_ests, N vector, vector of prediction point estimates  
    :param y_tests, N vector, vector of test responses
    :param rhos, M vector, threshold values
    
    :returns cap_metrics, M x 4 matrix, matrix holding abstention degree, capacity classification
                                        accuracy capacity F score, capacity AUC score
    """

    cap_metrics = np.zeros((rhos.shape[0], 4))

    for i in range(rhos.shape[0]):
        try:
            cap_metrics[i, :] = predictive_metrics(prob_ests, y_ests, y_test, rho[i])
        except ThresholdError:
            return cap_metrics[0:i, ]
        
    return cap_metrics

def max_threshold(prob_ests):
    """
    Computes the maximum value of the abstention threshold
    
    :param prob_ests, N vector, vector of predictive probability point estimates

    :returns max_thresh, float[0. 1], threshold value that removes all data
    """
    
    # Work out the two possiblities
    thresh1 = 0.5 - np.min(prob_ests)
    thresh2 = np.max(prob_ests) - 0.5
    
    # Find the maximum threshold value
    if thresh2 > thresh1:
        max_thresh = thresh2
    else:
        max_thresh = thresh1
        
    return max_thresh

def inspect_metrics(accuracy_ax, f_score_ax, auc_ax, cap_metrics):
    """
    Produces plots of the capacity accuracy, capacity F score and capacity AUC scores vs
    the degree of abstention
    
    :param cap_metrics, M x 4 matrix, matrix holding abstention degree, capacity classification
                                        accuracy capacity F score, capacity AUC score
    """
    
    
    accuracy_ax.plot(cap_metrics[:, 0], cap_metrics[:, 1], label = 'Capacity Accuracy Score')
    accuracy_ax.set_xlabel('Degree of Abstention', fontsize = 15)
    accuracy_ax.set_ylabel('Capacity Classification Accuracy', fontsize = 15)
    accuracy_ax.set_ylim([-0.05, 1.05])
    
    f_score_ax.plot(cap_metrics[:, 0], cap_metrics[:, 2], label = 'Capacity F Score')
    f_score_ax.set_xlabel('Degree of Abstention', fontsize = 15)
    f_score_ax.set_ylabel('Capacity F Score', fontsize = 15)
    f_score_ax.set_ylim([-0.05, 1.05])
    
    auc_ax.plot(cap_metrics[:, 0], cap_metrics[:, 3], label = 'Capacity AUC Score')
    auc_ax.set_xlabel('Degree of Abstention', fontsize = 15)
    auc_ax.set_ylabel('Capacity AUC Score', fontsize = 15)
    auc_ax.set_ylim([-0.05, 1.05])