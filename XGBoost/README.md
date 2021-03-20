Boosting:

Boosting is yet another ensemble machine learning technique which is designed to improve or boost the performance of a no. of weak learning models working together.

The weak Classifier:

A weak classifier is a model which can’t produce a sable result on it’s own but performs better than a random guess.

Here is the Mathematical representation of a boosting classifier<br>
                                               ![formula](https://user-images.githubusercontent.com/58946705/111885101-a7c5a580-89eb-11eb-8dee-fd71bc899a2d.jpg)

Here, **T** is the no. of classifiers, **ft** is a weak learner and **x** is the data input. It returns the class

 It can be depicted by the below diagram:
 ![flowchart](https://user-images.githubusercontent.com/58946705/111885187-2a4e6500-89ec-11eb-919d-a9f260f2487c.jpg)


### XGBoost
XGBoost improves the gradient boosting method even further.
It was developed by Tianqi Chen in C++ but now has interfaces for Python, R, Julia.

XGBoost's objective function is the sum of loss function evaluated over all the predictions and a regularisation function for all predictors ( 𝑗  trees). In the formula  𝑓𝑗  means a prediction coming from the  𝑗𝑡ℎ  tree.


![xgbf](https://user-images.githubusercontent.com/58946705/111885430-be6cfc00-89ed-11eb-825f-7d769a428a68.jpg)



Unlike the other tree-building algorithms, XGBoost doesn’t use entropy or Gini indices. Instead, it utilises gradient (the error term) and hessian for creating the trees.
