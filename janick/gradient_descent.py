import numpy as np
import matplotlib.pyplot as plt

class GDRegressor:

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.history = []
        self.optimal_w = 0

    def fit(self, X, y):
        X_b = np.c_[np.ones((100, 1)), X]
        self.gradient_descent(X_b, y, w)

    # RSS cost function
    def RSS(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return (1/(2*m)) * np.sum(errors**2)

    # Gradient of the RSS cost function
    def RSS_gradient(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return (1/m) * X.T.dot(errors)

    # Gradient descent
    def gradient_descent(self, X, y, w):
        for i in range(self.max_iter):
            w = w - self.learning_rate * self.RSS_gradient(X, y, w)
            self.history.append((w.copy(), self.RSS(X, y, w)))
            self.optimal_w = w
        return

if __name__ == '__main__':
    np.random.seed(0)
    X = 2 * np.random.rand(100, 2)
    y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

    w = np.random.randn(3)
    learning_rate = 0.1
    max_iter = 20

    model = GDRegressor(learning_rate, max_iter)

    model.fit(X, y)

    # Plot the cost function history
    costs = [h[1] for h in model.history]
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.show()

    print(f'Optimal parameters: {model.optimal_w}')
