# Perceptron
# Maths behind Perceptron Training
# -------- MODEL AND HELPER FUNCTIONS ---------
# Sigmoid function is an activation function (denoted as sigma(z)). The output of the sigma(z) belongs to the range 0 to 1. 
# 0 means - highly negative input and 1 means - highly positive input 
# This is useful as an activation function when one is interested in probability mapping rather than precise values of input parameter t.

import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=500):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy