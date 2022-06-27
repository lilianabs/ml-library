import numpy as np


def accuracy(predictions, y_test):
    """Computes the accuracy of a classification model

    Args:
        predictions (np.array): Predictions of the classification model.
        y_test (np.array): Actual labels of the data.

    Returns:
        float: Accuracy of the predictions.
    """
    return np.sum(predictions == y_test) / len(y_test)


def root_mean_squared_error(predictions, y_test):
    return np.sqrt(np.sum((predictions - y_test) ** 2) / len(predictions))
