# Ridge Regression

## &nbsp; Introduction
- Ridge regression also known as L2 regularization is a special case of Tikhonov regularization in which all parameters are regularized equally.
- It was first introduced by Hoerl and Kennard in 1970. 
- It is a technique which is used for analyzing multiple regression data that suffer from multicollinearity.
 

## &nbsp; Working of Ridge Regression
- The objective of ridge regresion is to minimize the loss function plus the sum of square of the magnitude of coefficients.
- The Ridge regression puts a constraint on the sum of suqare of values of the model parameters,the sum has to be less than a fixed value (upper bound).
- In order to do so, it applies a shrinking (regularization) process where it penalizes the coefficients of the regression variables. 
- The goal of this process is to minimize the prediction error.


## &nbsp; Ridge Regression Cost Function
<p align="center">
  <img src="https://github.com/divyanshu887/helper/blob/main/ridge_cost_function.jpeg">
</p>

## &nbsp; Ridge Regression Constrain
<p align="center">
  <img src="https://github.com/divyanshu887/helper/blob/main/ridge_constrain.jpeg">
</p>

&nbsp;&nbsp;where,
- c is the upper bound
- n is number of training examples
- D is the number of features 
- λ denotes the amount of shrinkage.

## &nbsp; Different cases for tuning values of lambda.
- λ = 0 implies all features are considered and it is equivalent to the linear regression where only the residual sum of squares is considered to build a predictive model
- λ = ∞ implies no feature is considered i.e, as λ closes to infinity it eliminates more and more features
- 0 < λ < ∞ : We get weights between 0 and that of simple linear regression
- The bias increases as λ increases.
- The variance decreases as λ increases.

## &nbsp; Advantage of Ridge Regression
- It avoids overfitting. 
- It trades variance for bias. 
- It generally works well even in presence of highly correlated features as it will include all of them in the model but the coefficients will be distributed among them depending on the correlation.

## &nbsp; Disadvantage of Ridge Regression
- It cannot shrink coefficients to exactly zero, which indicates that there is no feature selection.   
- Its model interpretability is low
- It increases bias

## &nbsp; Reference 
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Ridge regression - Wikipedia](https://en.wikipedia.org/wiki/Ridge_regression)
- [How to Develop LASSO Regression Models by Jason Brownlee](https://machinelearningmastery.com/ridge-regression-with-python/)
- [Regularization Part 1: Ridge (L2) Regression by StatQuest with Josh Starmer](https://youtu.be/Q81RR3yKn30)
