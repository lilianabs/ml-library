import sys
import pytest

from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ml_library.knn import KNN
from ml_library.metrics import accuracy
from ml_library.metrics import root_mean_squared_error

sys.path.insert(0, "../ml_library/")


@pytest.fixture
def data_classification():
    # Load Iris dataset
    return load_iris()


@pytest.fixture
def data_regression():
    # Load California Housing dataset
    return fetch_california_housing()


def test_load_iris_data(data_classification):

    X = data_classification.data
    y = data_classification.target

    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_california_housing(data_regression):

    X = data_regression.data
    y = data_regression.target

    assert X.shape == (20640, 8)
    assert y.shape == (20640,)


def test_classification(data_classification):

    X = data_classification.data
    y = data_classification.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    acc = accuracy(predictions, y_test)

    assert acc > 0.90


def test_regression(data_regression):

    X = data_regression.data
    y = data_regression.target

    X_train, X_test, y_train, y_test = train_test_split(
        X[:500], y[:500], test_size=0.2, random_state=1
    )

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    rmse = root_mean_squared_error(predictions, y_test)

    assert rmse < 1.10
