import numpy as np


class LogisticRegression:
    """Logistic regression algorithm for classifying instances
    of two classes
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """Initializes a Logistic Regression object"""
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Finds the best parameters (weights and bias)
           for the logistic regression model using the
           algorithm gradient descent

        Args:
            X (np.array): Features
            y (np.array): Target
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent algorithm
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.bias * db

    def predict(self, X):
        """Predicts the target for the given
           features.

        Args:
            X (np.array): Features

        Returns:
            np.array: Predictions
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if y > 0.5 else 0 for y in y_predicted]

        return y_predicted_cls

    def _sigmoid(self, x):
        """Computes the sigmoid function

        Args:
            x (float): input

        Returns:
            float: result of the sigmoid function
        """
        return 1 / (1 + np.exp(-x))
