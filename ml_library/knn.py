import numpy as np

from collections import Counter
from .base_class import BaseML
from .utils import euclidean_distance


class KNN(BaseML):
    """K-nearest neighboors algorithm for
    classification and regression.
    """

    def __init__(self, k=3, regression=True):
        """Initializes a KNN object

        Args:
            k (int, optional): Number of k-nearest neighboors.
                               Defaults to 3.
            regression (bool, optional): Regression or
                                         classification
                                         algorithm.
                                         Defaults to True.
        """
        super().__init__()
        self.k = k
        self.regression = regression

    def fit(self, X, y):
        """Loads the dataset in memory for making predictions.

        Args:
            X (np.array): Features
            y (np.array): Target
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predicts the target.

        Args:
            X (np.array): Features for the prediction.

        Returns:
            list(float): List with the predictions.
        """
        if self.regression:
            predicted_values = [self._predict_regression(x) for x in X]
            return predicted_values
        else:
            predicted_labels = [self._predict_classification(x) for x in X]
            return predicted_labels

    def _find_k_nearest_neighboors(self, x):
        """Finds k-nearest neigboors of x

        Args:
            x (np.array): Single data point.

        Returns:
            list(float): K-nearest neighboors of x.
        """
        # compute distances
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest neighbors and labels
        k_indices = np.argsort(distance)[: self.k]
        k_nearest_neighboors = [self.y_train[i] for i in k_indices]
        return k_nearest_neighboors

    def _predict_classification(self, x):
        """Computes predictions for classification.

        Args:
            x (np.array): Single data point.

        Returns:
            int: Most common class of the k-nearest neighboors.
        """
        k_nearest_labels = self._find_k_nearest_neighboors(x)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _predict_regression(self, x):
        """Computes predictions for regression.

        Args:
            x (np.array): Single data point.

        Returns:
            float: Average target value of the k-nearest neighboors.
        """
        k_nearest_values = self._find_k_nearest_neighboors(x)
        average_value = np.average(k_nearest_values)
        return average_value
