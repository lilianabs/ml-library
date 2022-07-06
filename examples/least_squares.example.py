import sys
import os
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

current_dir = os.getcwd()
sys.path.append(current_dir)

path = Path(current_dir)
a = str(path.parent.absolute())
sys.path.append(a)

from ml_library.least_squares import LeastSquares  # noqa: E402
from ml_library.metrics import root_mean_squared_error  # noqa: E402


def test_least_squares():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    regressor = LeastSquares()

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)

    print("RMSE of the least squares model: {}".format(rmse))


if __name__ == "__main__":
    """This is executed when run from the command line"""
    test_least_squares()
