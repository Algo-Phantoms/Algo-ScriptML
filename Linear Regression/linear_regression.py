import numpy as np

class LinearRegression():
""" This is an implementation of a Linear Regression Algorithm."""
    def __init__(self, X, y, learning_rate=0.03, n_iterations=1500):
        """Parameters:

        n_iterations: float
            The number of training iterations the algorithm will tune the weights for.
        learning_rate: float
            The step length that will be used when updating the weights.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones((self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.parameters = np.zeros((self.n_features + 1, 1))
        self.coeficiant_ = None
        self.intercept_ = None

    def fit(self):
        for _ in range(self.n_iterations):
            self.parameters = self.parameters - (self.learning_rate/self.n_samples) * self.X.T @ (self.X @ self.parameters - self.y)
        self.intercept_ = self.parameters[0]
        self.coeficiant_ = self.parameters[1:]
        return self

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones((n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]
        y_pred = X @ self.parameters
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        return score

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) / np.std(X, 0))) @ self.parameters
        return y

    def get_parameters(self):
        return self.parameters