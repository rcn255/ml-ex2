import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LinearRegression:

    def __init__(self, alpha=0.01, num_iters=1000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.theta = None
        self.cost_history = []
        self.theta_history = []

    # Cost function (RSS)
    def J(self, X, y, theta=None):
        if theta is None:
            theta = self.theta
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        return cost

    # Gradient of the cost function
    def gradient(self, X, y):
        m = len(y)
        predictions = X.dot(self.theta)
        errors = predictions - y
        grad = (1 / m) * X.T.dot(errors)
        return grad

    def gradient_descent(self, X, y):
        m = len(y)
        self.cost_history = np.zeros(self.num_iters)
        self.theta_history = np.zeros((self.num_iters, X.shape[1]))

        # Initialize theta with zeros or random values
        self.theta = np.random.randn(X.shape[1], 1)

        for i in range(self.num_iters):
            grad = self.gradient(X, y)  # Compute the gradient 
            self.theta -= self.alpha * grad  # Update theta
            self.cost_history[i] = self.J(X, y)  # Record the cost
            self.theta_history[i] = self.theta.T  # Record the theta

        return self

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        self.gradient_descent(X_b, y)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
        return X_b.dot(self.theta)
    

    def plot_gradient_descent(self, X, y):
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$J(\theta)$')
        
        # Plot the cost function
        theta_vals = np.linspace(-10, 10, 1000)
        J_vals = np.array([self.J(X, y, np.array([[theta]])) for theta in theta_vals])
        ax.plot(theta_vals, J_vals, 'b-', linewidth=2)

        # Initialize plot elements
        line, = ax.plot([], [], 'r', linewidth=2)
        point, = ax.plot([], [], 'ro')

        def update(i):
            theta_i = self.theta_history[:i+1, 0]
            cost_i = [self.J(X, y, np.array([[theta]])) for theta in theta_i]
            line.set_data(theta_i, cost_i)
            point.set_data([theta_i[-1]], [cost_i[-1]])
            return line, point

        ani = FuncAnimation(fig, update, frames=range(self.num_iters), blit=True, interval=100, repeat=False)
        plt.show()

    


# Synthetic data
"""
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

model = LinearRegression(alpha=0.1, num_iters=100)
model.fit(X, y)
model.plot_gradient_descent(X, y)
"""

