import numpy as np
import matplotlib.pyplot as plt

class knnRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        # Distance between x and all data points in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        
        # Find the k-nearest neighbors and their values
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = [self.y_train[i] for i in k_indices]
        
        # mean of the k-nearest neighbors
        return np.mean(k_nearest_values)


