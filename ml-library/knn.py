from base_class import BaseML

class KNN(BaseML):

    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        print("fit")
        pass

    def predict(self, X):
        print("predict")
        pass