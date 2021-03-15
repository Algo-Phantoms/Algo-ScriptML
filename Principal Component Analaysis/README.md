# Principal Component Analysis

**What is PCA?**

In simple words, PCA is a method of obtaining important variables (in form of components) from a large set of variables available in a data set. It extracts low dimensional set of features by taking a projection of irrelevant dimensions from a high dimensional data set with a motive to capture as much information as possible. With fewer variables obtained while minimising the loss of information, visualization also becomes much more meaningful. PCA is more useful when dealing with 3 or higher dimensional data.

It is always performed on a symmetric correlation or covariance matrix. This means the matrix should be numeric and have standardized data.

# What are the Principal Components?

A principal component is a normalized linear combination of the original predictors in a data set.

Let’s say we have a set of predictors as X¹, X²...,Xp

The principal component can be written as:

Z¹ = Φ¹¹X¹ + Φ²¹X² + Φ³¹X³ + .... +Φp¹Xp

* Z¹ is first principal component
* X¹..Xp are normalized predictors. Normalized predictors have mean equals to zero and standard deviation equals to one.

First principal component is a linear combination of original predictor variables which captures the maximum variance in the data set. It determines the direction of highest variability in the data. Larger the variability captured in first component, larger the information captured by component. No other component can have variability higher than first principal component.

**The first principal** component results in a line which is closest to the data i.e. it minimizes the sum of squared distance between a data point and the line.

---------------------------------------------------------------------------
Similarly, we can compute the second principal component also.

 
**Second principal component (Z²)** is also a linear combination of original predictors which captures the remaining variance in the data set and is uncorrelated with Z¹. In other words, the correlation between first and second component should is zero. It can be represented as:

Z² = Φ¹²X¹ + Φ²²X² + Φ³²X³ + .... + Φp2Xp

If the two components are uncorrelated, their directions should be orthogonal.

# Conclusion:

Thus **Principal Component Analysis** is used to remove the redundant features from the datasets without losing much information.
* These features are low dimensional in nature.The first component has the highest variance followed by second, third and so on.
* PCA works best on data set having 3 or higher dimensions.
* Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data.
