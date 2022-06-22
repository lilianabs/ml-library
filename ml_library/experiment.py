from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()

X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNN()
knn.fit(X_train, y_train)
knn.predict(X_test)
