# LOWESS REGRESSION

## Introduction

LOWESS are non-parametric regression methods that combine multiple regression models in a k-nearest-neighbor-based meta-model. They address situations in which the classical procedures do not perform well or cannot be effectively applied without undue labor. LOWESS combines much of the simplicity of linear least squares regression with the flexibility of nonlinear regression. It does this by fitting simple models to localized subsets of the data to build up a function that describes the variation in the data, point by point.

## Procedure

A linear function is fitted only on a local set of points delimited by a region, using weighted least squares. The weights are given by the heights of a kernel function (i.e. weighting function) giving:

▪ more weights to points near the target point x0 whose response is being estimated

▪ less weight to points further away

We obtain then a fitted model that retains only the point of the model that are close to the target point (x0). The target point then moves away on the x axis and the procedure repeats for each point.

## Advantages

▪ Allows us to put less care into selecting the features in order to avoid overfitting.

▪ Does not require specification of a function to fit a model to all of the data in the sample.

▪ Only a Kernel function and smoothing / bandwidth parameters are required.

▪ Very flexible, can model complex processes for which no theoretical model exists.

▪ Considered one of the most attractive of the modern regression methods for applications that fit the general framework of least squares regression but which have a complex deterministic structure.

## Disadvantages

▪ Requires to keep the entire training set in order to make future predictions.

▪ The number of parameters grows linearly with the size of the training set.

▪ Computationally intensive, as a regression model is computed for each point.

▪ Requires fairly large, densely sampled data sets in order to produce good models. This is because LOWESS relies on the local data structure when performing the local fitting.

## References

▪ https://xavierbourretsicotte.github.io/loess.html
