# Lasso Regression

## Introduction

- LASSO stands for Least Absolute Shrinkage and Selection Operator, also known as L1 regularization.
- It was first introduced in 1996 by Statistics Professor Robert Tibshirani.
- It is a statistical formula which performs feature selection and regularization of data models


### Working of Lasso Regression 
- The objective of lasso regresion is to minimize the loss function plus the sum of absolute value of the magnitude of weights  
- The LASSO regression puts a constraint on the sum of the absolute values of the model parameters,the sum has to be less than a fixed value (upper bound). 
- In order to do so, it applies a shrinking (regularization) process where it penalizes the coefficients of the regression variables shrinking some of them to zero. 
- After shrinkage,it does features selection process in which variables that still have a non-zero coefficient are selected to be part of the model. 
- The goal of this process is to minimize the prediction error.  
 
### Lasso Regression Cost Function     
<p align="center">
  <img src="https://miro.medium.com/max/431/1*PJav7bnRliTqNaeDVOjLWQ.gif">
 </p>
 
 #### Lasso Regression Constraint
<p align="center">
  <img src="https://miro.medium.com/max/116/1*Zstaco2-yAYBmHDCbsQstQ.gif">
</p>


where,
- c is the upper bound
- n is number of training examples
- D is the number of features 
- λ denotes the amount of shrinkage.


### Different cases for tuning values of lambda.
- λ = 0 implies all features are considered and it is equivalent to the linear regression where only the residual sum of squares is considered to build a predictive model
- λ = ∞ implies no feature is considered i.e, as λ closes to infinity it eliminates more and more features
-When 0 < λ < ∞ : We get weights between 0 and that of simple linear regression
- The bias increases with increase in λ
- variance increases with decrease in λ





### Advantage of Lasso Regression 
- It avoids over fitting by eliminating the lesser significant and irrelevant data 
- It can provide a very good prediction accuracy 

### Disadvantage of Lasso Regression 
- Selected features will be highly biased.
- For n<<p (n-number of data points, p-number of features), LASSO selects at most n features.
- It can not do group selection i.e. it will select only one feature from a group of correlated features, the selection is arbitrary in nature.


### How to use
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
"""
 lamda: float
       The factor that will determine the amount of regularization and feature shrinkage. 
 epoch: float
        The number of training iterations the algorithm will tune the weights for.
 learning_rate: float
        The step length that will be used when updating the weights.
 """

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

### References 
- [sklearn.linear_model.Lasso ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [How to Develop LASSO Regression Models by Jason Brownlee](https://machinelearningmastery.com/lasso-regression-with-python/)

- [Regularization Part 2: Lasso (L1) Regression by StatQuest with Josh Starmer
](https://www.youtube.com/watch?v=NGf0voTMlcs)

