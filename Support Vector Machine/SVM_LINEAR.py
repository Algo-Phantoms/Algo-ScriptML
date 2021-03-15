class SVM:

    def __init__(self, alpha=0.001, lambda1=0.01, epochs=1000):
        self.alpha = alpha
        self.lambda1 = lambda1
        self.epochs = epochs
        self.weights = None
        self.b = None

    def fit(self, X, y):
        cols, rows = X.shape
        y1 = np.where(y <= 0, -1, 1)
        self.weights = np.random.randn(rows)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(y1)):
                if y1[i] * (np.dot(X[i], self.weights) - self.b) >= 1:
                    self.weights -= self.alpha * (2 * self.lambda1 * self.weights)
                else:
                    self.weights -= self.alpha * (2 * self.lambda1 * self.weights - y1[i] * X[i])
                    self.b -= self.alpha * y1[i]

    def predict(self, X):
        predict_ = np.dot(X, self.weights) - self.b
        for i in range(len(predict_)):
            if predict_[i] == -1:
                predict_[i] = 0
        return np.sign(predict_)