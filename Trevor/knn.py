import numpy as np
import math
import scipy.spatial


class KNNRegressor:

    def __init__(self, n_neighbors=1, strategy='average'):
        self.nn = n_neighbors
        self.strategy = strategy
        self.KDTree = None
        self.y = None

    def fit(self, X, y):
        #X = X.to_numpy()
        y = y.to_numpy()
        self.KDTree = scipy.spatial.KDTree(X, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        self.y = y

    def predict(self, X):
        #X = X.to_numpy()
        predictions = []

        for i in range(X.shape[0]):
            # Find the nearest neighbors for the current test sample
            distances, indices = self.KDTree.query(X[i], k=self.nn)

            # Retrieve the corresponding values from y
            nearest_values = self.y[indices]

            if self.strategy == 'average':
                # Compute the average value of the nearest neighbors
                average = np.mean(nearest_values)
                predictions.append(average)

        return np.array(predictions)
