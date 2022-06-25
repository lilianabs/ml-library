import numpy as np


def euclidean_distance(x1, x2):
    """Computes the euclidean distance between two points

    Args:
        x1 (np.array): Fist point in the euclidean plane.
        x2 (np.array): Second point in the euclidean plane.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


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
