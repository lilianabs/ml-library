import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

current_dir = os.getcwd()
sys.path.append(current_dir)

path = Path(current_dir)
a = str(path.parent.absolute())
sys.path.append(a)

from ml_library.logistic_regression import LogisticRegression  # noqa: E402
from ml_library.metrics import accuracy  # noqa: E402

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

classifier = LogisticRegression(lr=0.0001, n_iters=1000)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_valid)
acc = accuracy(y_valid, predictions)

print("Logistic regression accuracy: ", acc)
