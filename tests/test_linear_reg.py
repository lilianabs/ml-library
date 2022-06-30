import sys
import pytest

from sklearn import datasets
from sklearn.model_selection import train_test_split
from ml_library.metrics import root_mean_squared_error
from ml_library.linear_regression import LinearRegression

sys.path.insert(0, "../ml_library/")


@pytest.fixture
def data():

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    return X, y


def test_predictions(data):

    X, y = data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    lr = LinearRegression(lr=0.01)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    rmse = root_mean_squared_error(predictions, y_test)
    assert rmse < 20.0
