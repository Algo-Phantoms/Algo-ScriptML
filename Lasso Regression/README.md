# Lasso Regression
- LASSO stands for Least Absolute Shrinkage and Selection Operator. 
- It is a type of linear regression which uses shrinkage where data values shrink toward central point to avoid overfitting.

### Loss Function for Lasso Regression -
<p align="center">
  <img src="https://geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-94965176c0038cdf5055dc2af5e7ec8e_l3.svg">
</p>

#### Different cases for tuning values of lambda.
- If lambda is set to be 0,   Lasso Regression equals Linear Regression.
- If lambda is set to be infinity, all weights are shrunk to zero

## How to use ?

```python
from Lasso_Regression import LassoRegression

"""
X.shape = (m,n)
Y.shape = (m,)

where,
    m = number of training examples
    n = number of features 
"""

# Create LassoRegression Object
model = LassoRegression( epoch = 1000, learning_rate = 0.01, lamda = 1 )

# Call fit method
model.fit(X_train, Y_train)

# Predict Method
y_pred = model.predict(X_test)

"""
y_pred.shape = (m,1)
where,
    m = number of training examples
"""
```
