import sys
import pytest

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ml_library.least_squares import LeastSquares
from ml_library.metrics import root_mean_squared_error

sys.path.insert(0, "../ml_library/")


@pytest.fixture
def data():
    return fetch_california_housing()


def test_least_squares(data):
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    regressor = LeastSquares()

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    assert rmse < 0.75
