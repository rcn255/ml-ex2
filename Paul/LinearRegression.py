import numpy as np
class LinearRegression:

    def __init__(self, alpha=0.01, num_iters=1000):
        self.alpha = alpha
        self.num_iters = num_iters
        self.theta = None
        self.cost_history = []

    # Cost function (RSS)
    def J(self, X, y):
        m = len(y)
        predictions = X.dot(self.theta)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        return cost

    def gradient_descent(self, X, y):
        m = len(y)
        self.cost_history = np.zeros(self.num_iters)
        self.theta = np.random.randn(X.shape[1], 1)

        for i in range(self.num_iters):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = (1 / m) * X.T.dot(errors)  # Gradient vector
            self.theta = self.theta - self.alpha * gradient  # Update theta
            self.cost_history[i] = self.J(X, y)  # Save cost for plotting

            # print out convergence information
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {self.cost_history[i]}")  

        return self

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        y = y.reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X] 
        self.gradient_descent(X_b, y) 

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        return X_b.dot(self.theta)
    