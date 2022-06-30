import numpy as np


class LinearRegression:
    """Linear regression with gradient descent"""

    def __init__(self, lr=0.001, n_iters=1000):
        """Initializes a Linear Regression object

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            n_iters (int, optional): Number of iterations for
                                     gradient descent. Defaults to 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fits a regression line using gradient descent algorithm.

        Args:
            X (np.array): Features
            y (np.array): Target
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """Predicts values for X

        Args:
            X (np.array): Features for prediction.

        Returns:
            np.array: Predictions for X.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
