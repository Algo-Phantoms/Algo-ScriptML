import numpy as np

class PCA:
    """
    PCA: mathematical technique used for dimensionality reduction
    Attrubutes:
    
    array (list): A matrix of elements
    """

    def __init__(self, array):
        self.arr = array

    def calculate(self):
        self.arr = np.array(self.arr)
        # Calculate mean
        arr_mean = np.mean(self.arr.T, axis = 1)
        # Scale the columns by subracting the column mean
        arr_scale = self.arr - arr_mean
        # Calculate the co-variance of the scaled transpose
        arr_cov = np.cov(arr_scale.T)
        # get the eigen values and vectors
        values, vectors = np.linalg.eig(arr_cov)
        # Matrix after applying PCA
        P = vectors.T.dot(arr_scale.T)
        return P.T


"""
Test case

arr = [
    [1, 2],
    [3, 4],
    [5, 6]
]

pca = PCA(arr)
print('Principal Component Analysis of the given array\n')
print(pca.calculate())

"""

"""
Solution

Principal Component Analysis of the given array

[[-2.82842712  0.        ] 
 [ 0.          0.        ] 
 [ 2.82842712  0.        ]]
"""
