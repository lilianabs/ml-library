from ml_library.knn import KNN
from ml_library.utils import accuracy
from ml_library.utils import root_mean_squared_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


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
