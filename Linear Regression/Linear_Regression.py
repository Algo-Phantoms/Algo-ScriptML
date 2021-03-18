import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):
        """
        Linear regression
        Arguments - alpha (by default .001)
                    epochs(by default 100)
                    plot : bool - plot the cost vs epoch graph
        function -  fit : (x, y) - train the data
                    predict : (x) - predict the new y using previus trained model
                    this module also normalize data brfore training that increase training efficiency
        """
        self.costs = []
        self.b = 0
        self.w = []

    @classmethod
    def forward(x, w, b):
        y_pred = np.dot(x, w) + b
        return y_pred

    @classmethod
    def compute_cost(y_pred, y, n):
        cost = (1/(2*n)) * np.sum(np.square(y_pred - y))
        return cost

    @classmethod
    def backward(y_pred, y, x, n):
        # print(y_pred.shape, y.shape, x.shape)
        dw = (1/n) * np.dot(x.T, (y_pred-y))
        # print(dw.shape)
        db = (1/n) * np.sum((y_pred-y))
        return dw, db

    @classmethod
    def update(w, b, dw, db, lr):
        w -= lr*dw
        b -= lr*db
        return w, b

    @classmethod
    def normalize(df):
        result = df.copy()
        for feature_name in df.columns:
            mean = np.mean(df[feature_name])
            std = np.std(df[feature_name])
            result[feature_name] = (df[feature_name] - mean) / std
        return result

    @classmethod
    def initialize(m):
        w = np.random.normal(size=(m, 1))
        b = 0
        return (w, b)

    def fit(self, X, y, lr=.005, epochs=550):
        X = pd.DataFrame(X)
        X = self.normalize(X)
        y = np.reshape(np.array(y), (len(y), 1))
        n, m = X.shape
        w, b = self.initialize(m)
        w = w
        b = b
        self.w, self.b = self.initialize(m)
        for _ in range(epochs):
            y_pred = self.forward(X, self.w, self.b)
            cost = self.compute_cost(y_pred, y, n)
            dw, db = self.backward(y_pred, y, X, n)
            self.w, self.b = self.update(self.w, self.b, dw, db, lr)
            self.costs.append(cost)
        plt.plot(self.costs)

    def predict(self, X):
        X = pd.DataFrame(X)
        X = self.normalize(X)
        y_pred = self.forward(X, self.w, self.b)
        return y_pred
