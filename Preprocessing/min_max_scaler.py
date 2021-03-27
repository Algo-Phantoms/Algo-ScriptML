import numpy as np
np.seterr(divide='ignore', invalid='ignore')

class MinMaxScaler:
    
    def __init__(self, feature_range=(0,1), *args):
        '''
        >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        >>> scaler = MinMaxScaler()
        >>> print(scaler.fit(data))
        MinMaxScaler()
        >>> print(scaler.data_max_)
        [ 1. 18.]
        >>> print(scaler.transform(data))
        [[0.   0.  ]
        [0.25 0.25]
        [0.5  0.5 ]
        [1.   1.  ]]
        >>> print(scaler.transform([[2, 2]]))
        [[1.5 0. ]]
        '''
        try:
            if (feature_range[0] >= feature_range[1]):
                raise ValueError(f'Minimum of desired feature range must be smaller than maximum.')
            else:
                self._scale_min = feature_range[0]
                self._scale_max = feature_range[1]
                self._sample_size = None
                self._mins = None
                self._maxs = None
        except Exception as e:
            raise e

    def fit(self, x, *args):
        try:
            x = np.array(x, dtype=np.float64)
            self._sample_size = x.shape[1]
            self._mins = x.min(axis=0)
            self._maxs = x.max(axis=0)
            return self
        except Exception as e:
            raise e

    def transform(self, x, *args):
        try:
            x = np.array(x, dtype=np.float64)
            if self._maxs is None and self._mins is None:
                return f'NotFittedError: This MinMaxScaler instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            elif x.shape[1] != self._sample_size:
                return f'ValueError: X has {x.shape[1]} features, but MinMaxScaler is expecting {self._sample_size} features as input'
            else:
                x = (x - self._mins) / (self._maxs - self._mins)
                x = (x * (self._scale_max - self._scale_min)) + self._scale_min
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
            if self._maxs is None and self._mins is None:
                return f'NotFittedError: This MinMaxScaler instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            else:
                x = (x - self._scale_min) / (self._scale_max - self._scale_min)
                x = (x * (self._maxs - self._mins)) + self._mins
                return x
        except Exception as e:
            raise e        

    def __remove_outlier_by_zero(self, x):
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def __remove_outlier_by_one(self, x):
        return np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)

    @property
    def min_(self):
        _min = self._scale_min - (self._mins * self.scale_)
        return _min

    @property
    def scale_(self):
        _scale = (self._scale_max - self._scale_min) / (self._maxs - self._mins)
        _scale = self.__remove_outlier_by_one(_scale)
        return _scale

    @property
    def data_min_(self):
        return self._mins

    @property
    def data_max_(self):
        return self._maxs

    @property
    def data_range_(self):
        data_range = self._maxs - self._mins
        return data_range



