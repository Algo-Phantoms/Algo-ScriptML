import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class StandardScaler:
    
    def __init__(self, *args):
        '''
        >>> scaler = StandardScaler()
        >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        >>> print(scaler.fit(data))
        StandardScaler()
        >>> print(scaler.mean_)
        [0.5 0.5]
        >>> print(scaler.transform(data))
        [[-1. -1.]
        [-1. -1.]
        [ 1.  1.]
        [ 1.  1.]]
        >>> print(scaler.transform([[2, 2]]))
        [[3. 3.]]
        '''
        self._sample_size = None
        self._means = None
        self._stds = None

    def fit(self, x, *args):
        try:
            x = np.array(x, dtype=np.float64)
            self._sample_size = x.shape[1]
            self._means = np.mean(x, axis=0)
            self._stds = np.std(x, axis=0)
            return self
        except Exception as e:
            raise e

    def transform(self, x, *args):
        try:
            x = np.array(x, dtype=np.float64)
            if self._means is None and self._stds is None:
                return f'NotFittedError: This StandardScaler instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            elif x.shape[1] != self._sample_size:
                return f'ValueError: X has {x.shape[1]} features, but StandardScaler is expecting {self._sample_size} features as input'
            else:
                x = (x - self._means) / self._stds
                x = self.__remove_outlier_by_zero(x)
                return x
        except Exception as e:
            raise e

    def fit_transform(self, x, *args):
        try:
            self.fit(x)
            return self.transform(x)
        except Exception as e:
            raise e

    def inverse_transform(self, x, *args):
        try:
            x = np.array(x, dtype=np.float64)
            if self._means is None and self._stds is None:
                return f'NotFittedError: This StandardScaler instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            else:
                x = (x * self._stds) + self._means
                x = self.__remove_outlier_by_zero(x)
                return x
        except Exception as e:
            raise e

    def __remove_outlier_by_zero(self, x):
    	return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def mean_(self):
        return self._means

    @property
    def scale_(self):
        return self._stds






	

