import numpy as np
import math


class Lowess(object):
    def __init__(self):
        """
        Lowess regression (Locally weighted regression)
        Arguments - window (by default 10)
                    degree(by default 1)
                    use_matrix(by default False)
        function -  fit : (x, y) - train the data
                    predict : (x) - predict the new y using previus trained model
                    this module also normalize data brfore training that increase training efficiency
        """
        self.n_xx, self.min_xx, self.max_xx = None, None, None
        self.n_yy, self.min_yy, self.max_yy, self.degree = None, None, None, None
        self.window, self.use_matrix = None, None

    def _get_min_range(self, distances):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, self.window)
        if min_idx == n-1:
            return np.arange(n - self.window, n)

        min_range = [min_idx]
        while len(min_range) < self.window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)
    
    @staticmethod
    def tricubic(x):
        y = np.zeros_like(x)
        idx = (x >= -1) & (x <= 1)
        y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
        return y

    def _get_weights(self, distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = self.tricubic(distances[min_range] / max_distance)
        return weights

    def _normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def _denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy
    
    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val
    
    def fit(self, x, y, window = 10, use_matrix=False, degree=1):
        '''
        Some pre-defined checks
        1) length of x and y array should be same
        2) Window size cannot exceed the number of data points
        '''
        if x.shape[0] != y.shape[0]:
            raise ValueError("Found input variables with inconsistent numbers of samples: ["+str(x.shape[0])+","+str(y.shape[0])+"]")
        if x.shape[0] < window:
            raise Exception("Window size cannot exceed the number of data points")
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(x)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(y)
        self.degree = degree
        self.window = window
        self.use_matrix = use_matrix
        
    def predict(self, x):
        n_x = self._normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self._get_min_range(distances)
        weights = self._get_weights(distances, min_range)

        if self.use_matrix or self.degree > 1:
            wm = np.multiply(np.eye(self.window), weights)
            xm = np.ones((self.window, self.degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(self.degree + 1)])
            for i in range(1, self.degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self._denormalize_y(y)
    
    '''
    Here's an example for the usage of Lowess Regression.
    xx and yy are the input arrays
    
     xx = np.array([0.5578196, 2.0217271, 2.5773252, 3.4140288, 4.3014084,
                   4.7448394, 5.1073781, 6.5411662, 6.7216176, 7.2600583,
                   8.1335874, 9.1224379, 11.9296663, 12.3797674, 13.2728619,
                   14.2767453, 15.3731026, 15.6476637, 18.5605355, 18.5866354,
                   18.7572812])
    yy = np.array([18.63654, 103.49646, 150.35391, 190.51031, 208.70115,
                   213.71135, 228.49353, 233.55387, 234.55054, 223.89225,
                   227.68339, 223.91982, 168.01999, 164.95750, 152.61107,
                   160.78742, 168.55567, 152.42658, 221.70702, 222.69040,
                   243.18828])
    lowess=Lowess()
    
    lowess.fit(xx,yy,window=10, use_matrix=False, degree=1)
    
    for x in xx:
        y=lowess.predict(x)
        print(x,y)
    '''
