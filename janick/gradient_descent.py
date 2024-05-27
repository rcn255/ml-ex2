import numpy as np
import matplotlib.pyplot as plt

class GDRegressor:

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.history = []
        self.optimal_w = None

    def fit(self, X, y):
        w = np.random.randn(X.shape[1] + 1)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.gradient_descent(X_bias, y, w)

    # RSS cost function
    def RSS(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return (1/(2*m)) * np.sum(errors**2)

    # RMSE cost function for plot
    def RMSE(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return np.sqrt((1/m) * np.sum(errors**2))

    # Gradient of the RSS cost function
    def RSS_gradient(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return (1/m) * X.T.dot(errors)

    # Gradient descent
    def gradient_descent(self, X, y, w):
        for i in range(self.max_iter):
            old_w = w
            gradient_result = self.RSS_gradient(X, y, w)
            w = w - self.learning_rate * gradient_result
            print(f"Itaration {i}:\n New w: {w}\n Gradient: {gradient_result}")

            self.history.append((w.copy(), self.RMSE(X, y, w)))
            self.optimal_w = w
        return

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias.dot(self.optimal_w)

    # Plot the cost function history
    def plot_cost_history(self):
        costs = [h[1] for h in self.history]
        plt.plot(costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()

    # Plot parameter values over iterations
    def plot_parameter_history(self):
        num_params = len(self.history[0][0])
        num_features = num_params - 1  # Exclude bias term
        for param_idx in range(num_params):
            if param_idx == 0:
                # Bias term
                plt.plot([w[0][param_idx] for w in self.history], label='Bias')
            else:
                # Feature coefficients
                feature_idx = param_idx - 1
                plt.plot([w[0][param_idx] for w in self.history], label=f'θ{feature_idx}')
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Values Over Iterations')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    X = 2 * np.random.rand(100, 2)
    y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

    learning_rate = 0.1
    max_iter = 20

    model = GDRegressor(learning_rate, max_iter)

    model.fit(X, y)


    print(f'Optimal parameters: {model.optimal_w}')
