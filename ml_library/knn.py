from base_class import BaseML

class KNN(BaseML):

    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        pass

    def predict(self, X):
        pass