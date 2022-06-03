"""
Utility functions for running cross validated classification pipelines
"""
import numpy as np
from gproc.run_pipeline import run_pipeline

def run_cv(x_data, y_data, classifiers, splits=5):
    """
    Runs cross validation for provided x and y data, for each classifier.
    Provided splits indicates how many CV folds should be used.

    Parameters
    ----------
    x_data: total_n x D numpy array
    
    y_data : total_n x 1 numpy array
        array containing 1 or -1, the observations

    classifiers: dict
        key is classifier label
        value is "fit_function", which takes in training x and y and produces 
        funciton predicting y for new x
    
    splits : int
        number of cross validation folds to perform

    Returns
    ----------
    results: dict
        key is classifier label
        value is a list containing computed test statistics for each CV fold
    """
    # Make train/test splits
    total_n = x_data.shape[0]
    assert total_n == y_data.shape[0], 'X and y data do not match!'

    shuffled_full_ix = np.random.permutation(np.arange(total_n))
    split_ixs = np.array_split(shuffled_full_ix, splits)

    results = {label: [] for label in classifiers}
    for split in range(splits):
        # Get train/test split
        test_ix = split_ixs[split]
        train_ix = np.concatenate([
            split_ix
            for i, split_ix in enumerate(split_ixs)
            if i != split
        ])

        train_x = x_data[train_ix]
        train_y = y_data[train_ix]
        test_x = x_data[test_ix]
        test_y = y_data[test_ix]

        # Run each classifier
        for label, fit_classifier in classifiers.items():
            results[label].append(
                run_pipeline(fit_classifier, train_x, train_y, test_x, test_y)
            )

    return results