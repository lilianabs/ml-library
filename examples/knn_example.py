import sys
import os
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

current_dir = os.getcwd()
sys.path.append(current_dir)

path = Path(current_dir)
a = str(path.parent.absolute())
sys.path.append(a)

from ml_library.knn import KNN  # noqa: E402
from ml_library.metrics import accuracy  # noqa: E402
from ml_library.metrics import root_mean_squared_error  # noqa: E402


def test_classification():

    iris_data = load_iris()

    X = iris_data.data
    y = iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    acc = accuracy(predictions, y_test)
    print("Iris classification model accuracy: ", acc)


def test_regression():

    california_housing_data = fetch_california_housing()

    X = california_housing_data.data
    y = california_housing_data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    rmse = root_mean_squared_error(predictions, y_test)
    print("California Housing regression model rmse: ", rmse)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    test_classification()
    test_regression()
