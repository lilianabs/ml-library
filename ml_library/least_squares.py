from .base_class import BaseML
import numpy as np


class LeastSquares(BaseML):
    """Least squares algorithm for regression."""

    def __init__(self):
        """Initializes a LeastSquares object."""
        super().__init__()
        self.w = None

    def fit(self, X, y):
        """Computes the weights of the regression
            line using the normal equation.

        Args:
            X (np.array): Features
            y (np.array): Target
        """
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX_inv = np.linalg.inv(X.T.dot(X))
        self.w = XTX_inv.dot(X.T.dot(y))

    def predict(self, X):
        """Predicts the target for the given
           features.

        Args:
            X (np.array): Features

        Returns:
            np.array: Predictions
        """
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        return X.dot(self.w)
