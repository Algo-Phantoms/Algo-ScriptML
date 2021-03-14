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

# Normalization of variables is necessary in PCA.

**Reason**: 

The principal components are supplied with normalized version of original predictors. This is because, the original predictors may have different scales. For example: Imagine a data set with variables’ measuring units as gallons, kilometers, light years etc. It is definite that the scale of variances in these variables will be large.

Performing PCA on un-normalized variables will lead to insanely large loadings for variables with high variance. In turn, this will lead to dependence of a principal component on the variable with high variance. This is undesirable.

**Now let's understand this concept with help of example dataset using Scikit-learn and find out the components with maximum variance.**

# Step-1: 
* Import the required libraries
* REad the dataset.

# Step-2:
* Do the necessary data exploration part if required.

# Step-3:
* Then head towrds **Data Standardization** .
* **Standardization** refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data

# Step-4:
* Seperate the features from the labels
* Then using **StandardScaler** module of sklearn library, fit the seperated features.

# Step-5:
* Compute the important**Eigenvalue and EigenVectors**
* followed by covariance metarix.

# Step-6:
* **Select the principal components**. But before this...
* In order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.
    * Make a list of (eigenvalue, eigenvector) tuples
    * Sort the (eigenvalue, eigenvector) tuples from high to low
    * Visually confirm that the list is correctly sorted by decreasing eigenvalues

# Step-7:
* **Explained Variance** After sorting the eigenpairs, the next question is "how many principal components are we going to choose for our new feature subspace?" A useful measure is the so-called "explained variance," which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.
* And plot it to see the result.

# Step-8:
* Calculate the **Projection Matrix**
* The construction of the projection matrix that will be used to transform the analytics data onto the new feature subspace.
* **Projection Onto the New Feature Space** In this last step we will use the 7×2-dimensional projection matrix W to transform our samples onto the new subspace via the equation **Y=X×W**

# Step-9:
* **PCA** in Scikit-Learn.
* Use the PCA module from **sklearn.decomposition**
* And fit the model and plot it to see the component analysis.

# Conclusion:

Thus **Principal Component Analysis** is used to remove the redundant features from the datasets without losing much information.
* These features are low dimensional in nature.The first component has the highest variance followed by second, third and so on.
* PCA works best on data set having 3 or higher dimensions.
* Because, with higher dimensions, it becomes increasingly difficult to make interpretations from the resultant cloud of data.
