import sys
import pytest

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ml_library.logistic_regression import LogisticRegression
from ml_library.metrics import accuracy

sys.path.insert(0, "../ml_library/")


@pytest.fixture
def data_classification():
    # Load breast cancer dataset
    return load_breast_cancer()


def test_classification(data_classification):

    X = data_classification.data
    y = data_classification.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    classifier = LogisticRegression(lr=0.0001, n_iters=1000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    acc = accuracy(predictions, y_test)

    assert acc > 0.70
