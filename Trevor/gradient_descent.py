import numpy as np
import matplotlib.pyplot as plt

class GDRegressor:

    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.history = []
        self.optimal_w = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.zeros(X_bias.shape[1])
        self.gradient_descent(X_bias, y, w)

    def RSS(self, X, y, w):
        predictions = X @ w
        errors = predictions - y
        return (1/(2*len(y))) * np.sum(errors**2)

    def RSS_gradient(self, X, y, w):
        predictions = X @ w
        errors = predictions - y
        return (1/len(y)) * X.T @ errors

    def gradient_descent(self, X, y, w):
        for i in range(self.max_iter):
            gradient = self.RSS_gradient(X, y, w)
            if np.isnan(gradient).any() or np.isinf(gradient).any():
                raise ValueError("Gradient has NaN or Inf values. This may be caused by extreme values in your data or a too high learning rate.")
            w = w - self.learning_rate * gradient
            self.history.append((w.copy(), self.RSS(X, y, w)))
        self.optimal_w = w

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.optimal_w

    def plot_cost_history(self):
        costs = [h[1] for h in self.history]
        plt.plot(costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()



