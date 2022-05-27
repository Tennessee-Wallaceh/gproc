import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_real_data(dataset_path):
    """
    Takes in a single-space separated csv file and returns the entire
    dataset in two numpy arrays.

    Parameters
    ----------
    dataset_path: string
        relative path of dataset csv file

    Returns
    -------
    X: N x D numpy array
        Array containing N samples of D features
    y: N dimensional numpy vector
        numpy vector containg -1 and 1 targets for binary classification
    """

    df = pd.read_csv(dataset_path, sep=" ", header=None)
    y = df.pop(df.shape[1]-1).replace(0,-1).to_numpy()
    X = df.to_numpy()
    return X, y

def load_real_data_train_test(dataset_path, shuffle_seed=None):
    """
    Takes in a single-space separated csv file and returns a random train-test
    split of the data in separate numpy arrays.
    Parameters
    ----------
    dataset_path: string
        relative path of dataset csv file
    shuffle_seed: int
        optional random seed for reproducible shuffling

    Returns
    -------
    X_train: N x D numpy array
        Array containing N train samples of D features
    y_train: N dimensional numpy vector
        numpy vector containg N targets (-1 and 1) corresponding to the train data
    X_test: N x D numpy array
        Array containing N test samples of D features
    y_test: N dimensional numpy vector
        numpy vector containg N targets (-1 and 1) corresponding to the test data
    """

    df = pd.read_csv(dataset_path, sep=" ", header=None)
    if shuffle_seed==None:
        train_df, test_df = train_test_split(df, test_size=0.3)
    else:
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=shuffle_seed)
    y_train = train_df.pop(train_df.shape[1]-1).replace(0,-1).to_numpy()
    X_train = train_df.to_numpy()
    y_test = test_df.pop(test_df.shape[1]-1).replace(0,-1).to_numpy()
    X_test = test_df.to_numpy()

    return X_train, y_train, X_test, y_test
