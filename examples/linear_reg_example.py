import sys
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

current_dir = os.getcwd()
sys.path.append(current_dir)

path = Path(current_dir)
a = str(path.parent.absolute())

sys.path.append(a)

from ml_library.linear_regression import LinearRegression  # noqa: E402
from ml_library.metrics import root_mean_squared_error  # noqa: E402


X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# print(X_train.shape)
# print(y_train.shape)

lr = LinearRegression(lr=0.01)
lr.fit(X_train, y_train)
predictions = lr.predict(X_valid)

rmse = root_mean_squared_error(predictions, y_valid)
print("California Housing regression model rmse: ", rmse)

cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_valid, y_valid, color=cmap(0.9), s=10)
plt.plot(X, lr.predict(X), color="black", linewidth=2, label="Prediction")
plt.show()
