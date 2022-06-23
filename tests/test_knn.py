import sys
import pytest

sys.path.insert(0, "../")

from ml_library.knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

@pytest.fixture
def data():
    # Load Iris dataset
    return load_iris()

def test_load_iris_data(data):

    X = data.data
    y = data.target
    
    assert X.shape == (150, 4)
    assert y.shape == (150,)