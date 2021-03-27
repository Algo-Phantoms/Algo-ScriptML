# Stochastic gradient descent(SGD):
* stochastic gradient descent(SGD) is used for regression problems with **very large dataset(in millions).**
* SGD is same as gradient discent algorithm but have difference in optimization function.
* SGD is inspired by  Robbins–Monro algorithm of the 1950s.

## Basic idea behind SGD:
* SGD works as an iterative algorithm.
* It starts from a random point in dataset
* after that it tries to fit the training sets one by one.

## Working of SGD:

* first it initialize the theta(weights) to some random values.
* than it takes a dataset(a row) from training set and tries to fit perfectly and returns modefied theta(weights).
* that returned theta(weights) are applied over next dataset and tries to fit it perfectly and returns the theta.
* this loop runs untill last dataset.

## Intution behind SGD:
* first the algo shuffles the data, so that there should be no pattern can be seen firstly.
* than algo tries to fit nex data more accurately thn the previously.

## SGD function:

![](https://github.com/captainra1/images/blob/master/fn.png)

## Difference between SGD and Gradient descent(Batch descent):

![](https://github.com/captainra1/images/blob/master/diff.png)
 * gradient descent goes from 1 to m(no of datasets) in every iterations while SGD iterates one time over one dataset.

 ## Main advantages of SGD:
 * computational efficient.
 * model with large dataset can be trained eaisly.

 ## Main disadvantages of SGD:

* not effective over small datasets.

## Documentation:

```python
Stochastic_gradient_descent(learning_rate=0.1)
```
it takes the learning rate only, if you will not give it will be initialized to 0.1
```python
object.fit(X,y)
```
after making object of SGD type we have to call .fit() method in order to train model.

* X: feature dataset(shuffeled)
* y:label set(target set)(shuffled)

```python
object.predict(X_pred)
```
X_pred: features for which prediction is to be made.

## Example:

```python

algo=Stochastic_gradient_descent(learning_rate=0.03)
model=algo.fit(X,y)
predicted_value=model.predict(X_pred)
```
