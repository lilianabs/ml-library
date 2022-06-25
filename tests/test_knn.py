import sys
import pytest

from sklearn.datasets import load_iris

sys.path.insert(0, "../ml_library/")


@pytest.fixture
def data():
    # Load Iris dataset
    return load_iris()


def test_load_iris_data(data):

    X = data.data
    y = data.target

    assert X.shape == (150, 4)
    assert y.shape == (150,)
