"""
Logistic regression
Arguments - alpha (by default .001)
            epochs(by default 100)
            plot : bool - plot the cost vs epoch graph  
function -  fit : (x, y) - train the data
            predict : (x) - predict the new y using previus trained model
"""
import numpy as np
import matplotlib.pyplot as plt

class LogesticRegression:
    
    
    def __init__(self, alpha=0.001, epsilon=1e-13, epochs=100):
        self.w = []
        self.b = 0
        self.y_pred = 0
        self.Cost = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.epochs = epochs
        self.cache = []
        
    def _sigmoid(self, z):   
        sigma = 1/(1+np.e**(-z))
        return sigma
    
    def _cost(self, y_pred, y):
        n = len(y)             
        j = 1/n * np.sum(-y*np.log(self.y_pred) - (1 - y)*np.log(1-self.y_pred))
        return j
        
    def _forward(self, X):
        y_pred = self._sigmoid(np.dot(X, self.w) + self.b)
        y_pred = np.clip(y_pred, self.epsilon , 1-self.epsilon)
        return y_pred
        
    def _backward(self, X, y):
        n = len(X)
        dw = np.array(1/n * np.dot(X.T, (self.y_pred - y)))
        db = (1/n) * np.sum(y+self.y_pred-2*(y*self.y_pred))
        #db = (1/n) * np.sum((self.y_pred - y) / (self.y_pred * (1 - self.y_pred)))
        return (dw, db)
        
    def _update(self, dw, db):
        self.w -= self.alpha * dw
        self.b -= self.alpha * db  
        
    def fit(self, X, y, plot=True):
        y = np.reshape(np.array(y), (len(y), 1))
        n , m = X.shape

        self.w = np.random.normal(size = (m,1))
        
        for i in range(self.epochs):
            self.y_pred = np.array(self._forward(X))
            j = self._cost(self.y_pred, y)
            self.cache.append(j)
            dw, db = self._backward(X, y)
            self._update(dw, db)       
        
        if plot == True:
            self._plot()
        
        
    def predict(self, X):
        y_pred = self._forward(X)
        y_pred = [1 if i>0.5 else 0 for i in y_pred]
        self._plot()
        return y_pred
        
    def _plot(self):
        plt.plot(list(range(self.epochs)), self.cache, '-r')


    