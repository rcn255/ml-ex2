import numpy as np
import matplotlib.pyplot as plt

class GDRegressor:

    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.history = []
        self.optimal_w = None

    def fit(self, X, y):
        #print(f"X shape: {X.shape}")
        #print(f"y shape: {y.shape}")
        w = np.random.randn(X.shape[1] + 1)
        #print(f"w: {w}")
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        #print(X_bias)
        #print(f"X_bias shape: {X_bias.shape}")
        self.gradient_descent(X_bias, y, w)

    # RSS cost function
    def RSS(self, X, y, w):
        m = len(y)
        predictions = X.dot(w)
        errors = predictions - y
        return (1/(2*m)) * np.sum(errors**2)

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
            #print(f"gradient_result: {gradient_result}")
            print(f"Itaration {i}:\n New w: {w}\n Gradient: {gradient_result}")

            self.history.append((w.copy(), self.RMSE(X, y, w)))
            self.optimal_w = w
        return

    def predict(self, X):
        #print(self.optimal_w.shape)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        #print(X_bias)
        return X_bias.dot(self.optimal_w)

    def plot_cost_history(self):
        # Plot the cost function history
        costs = [h[1] for h in self.history]
        plt.plot(costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()

    def plot_parameter_history(self):
        # Plot parameter values over iterations
        num_params = len(self.history[0][0])  # Number of parameters
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
