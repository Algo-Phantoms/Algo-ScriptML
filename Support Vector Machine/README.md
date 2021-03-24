# Support Vector Machines

SVM is a computer algorithm used in supervised learning. It tunes math equations to give the most accurate answer possible for classification, regression, and outliers detection. There are specific types of SVMs you can use for particular machine learning problems, like **support vector regression** (SVR) which is an extension of **support vector classification** (SVC).

SVMs are different from other classification algorithms because of the way they choose the decision boundary that maximizes the distance from the nearest data points of all the classes. The decision boundary created by SVMs is called the **maximum margin classifier** or the **maximum margin hyper plane**.
![](data:image/png;base64,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)


A simple linear SVM classifier works by making a straight line between two classes. That means all of the data points on one side of the line will represent a category and the data points on the other side of the line will be put into a different category. This means there can be an infinite number of lines to choose from.
The dimension of hyperplane depends on the number of input features present. If the input features are 2, the hyperplane will be a line. If it is 3 then it will be a 2-D plane.
![](https://image.slidesharecdn.com/svm-140807035301-phpapp01/95/support-vector-machine-without-tears-6-638.jpg?cb=1407384107)

Basically you have some data points on a grid. You're trying to separate these data points by the category they should fit in, but you don't want to have any data in the wrong category. That means you're trying to find the line between the two closest points that keeps the other data points separated.

_So the two closest data points give you the support vectors you'll use to find that line. That line is called the **decision boundary**._

![](https://www.mathworks.com/help/stats/svmhyperplane.png)

_**Support Vectors** are the data points which are closer to the hyperplane and influence it's position and orientation._
Using these support vectors , we maximize the margin of the classifier. These points help us build the SVM. Deleting these points will change the position of the hyperplane.  

# Types of SVM
There are two different types of SVMs, each used for different things:

![](https://miro.medium.com/max/1418/0*gMrEyZgDJ_nIUcao.)
* **Simple SVM** : _Typically used for linear regression and classification problems._
* **Kernel SVM** : _Has more flexibility for non-linear data because you can add more features to fit a hyperplane instead of a two-dimensional space._

# Kernel

The SVM algorithm uses a set of mathematical functions that are defined as Kernels. Sometimes it is not possible to find a hyperplane or a linear decision boundary for some classification problems. If we project the data into a higher dimension from the original space, we may get a hyperplane in the projected dimension that helps to classify the data.

For example, it is impossible to find a line which separates the two classes in the input space, however, if we project the same data points or the input space into a higher dimension we could be able to classify the two classes using a _hyperplane_. Refer to the below example. Initially, it's hard to separate the two classes, however when it is projected into a higher dimension, then using hyperplane we could easily separate the two classes for classification.

![](https://qph.fs.quoracdn.net/main-qimg-6e7a32fdb8c29c808735c9755a60628b)

Thus Kernel helps to find a hyperplane in the higher dimensional space without increasing the computational cost. Usually, the computational cost will increase with the increase of dimensions.

Kernel trick is the function that transforms data into a suitable form. There are various types of kernel functions used in the SVM algorithm i.e. Polynomial, linear, non-linear, Radial Basis Function, etc. Here using kernel trick low dimensional input space is converted into a higher-dimensional space.

# Types of Kernel

## Linear
It is used when the data is linearly seperable. It is the most basic type of kernel, usually one dimensional in nature. It proves to be the best function when there are lots of features. The linear kernel is mostly preferred for text-classification problems as most of these kinds of classification problems can be linearly separated.

Here's the function that defines the linear kernel:
```python
f(X) = w^T * X + b
```
In this equation, **w** is the _weight vector_ that you want to _minimize_, **X** is the _data_ that you're trying to classify, and **b** is the _linear coefficient_ estimated from the training data. This equation defines the decision boundary that the SVM returns.

## Polynomial

It is a more generalized representation of the linear kernel. It is not as preferred as other kernel functions as it is **less efficient and accurate**.

Here's the function that defines the linear kernel:
```python
f(X1, X2) = (a + X1^T * X2) ^ b
```

This is one of the more simple polynomial kernel equations you can use. f(X1, X2) represents the polynomial decision boundary that will separate your data. X1 and X2 represent your data.

## RBF

One of the **most powerful** and commonly used kernels in SVMs. Usually the choice for **non-linear data**.
Here's the equation for an RBF kernel:
```python
f(X1, X2) = exp(-gamma * ||X1 - X2||^2)
```

In this equation, gamma specifies how much a single training point has on the other data points around it. ||X1 - X2|| is the dot product between your features.

## Sigmoid

More **useful in neural networks** than in support vector machines, but there are occasional specific use cases.
Here's the function for a sigmoid kernel:
```python
f(X, y) = tanh(alpha * X^T * y + C)
```
In this function, alpha is a weight vector and **C** is an _offset value_ to account for some mis-classification of data that can happen.

# Pros
1. Effective on datasets with multiple features, like financial or medical data.
2. Effective in cases where number of features is greater than the number of data points.
3. Uses a subset of training points in the decision function called support vectors which makes it memory efficient.
4. Different kernel functions can be specified for the decision function. You can use common kernels, but it's also possible to specify custom kernels.


# Cons

1. SVM algorithm is not suitable for large data sets.
2. SVM does not perform very well when the data set has more noise i.e. target classes are overlapping.
3. In cases where the number of features for each data point exceeds the number of training data samples, the SVM will underperform.
4. As the support vector classifier works by putting data points, above and below the classifying hyperplane there is no probabilistic explanation for the classification.

# Hyperparameters

The **Regularization parameter** (often termed as **C** parameter in python’s sklearn library) tells the SVM optimization how much you want to avoid misclassifying each training example.

_For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassified more points._

The **gamma parameter** defines how far the **influence of a single training example** reaches, with low values meaning ‘far’ and high values meaning ‘close’. 

_In other words, with low gamma, points far away from plausible separation lines are considered in the calculation for the separation line. Whereas high gamma means the points close to plausible lines are considered in the calculation._

# References
* [https://datascience.foundation/datatalk/basic-overview-of-svm-algorithm](https://datascience.foundation/datatalk/basic-overview-of-svm-algorithm)
* [https://dhirajkumarblog.medium.com/top-4-advantages-and-disadvantages-of-support-vector-machine-or-svm-a3c06a2b107](https://dhirajkumarblog.medium.com/top-4-advantages-and-disadvantages-of-support-vector-machine-or-svm-a3c06a2b107)
* [https://www.freecodecamp.org/news/svm-machine-learning-tutorial-what-is-the-support-vector-machine-algorithm-explained-with-code-examples/](https://www.freecodecamp.org/news/svm-machine-learning-tutorial-what-is-the-support-vector-machine-algorithm-explained-with-code-examples/)

