# Support Vector Machine

Support Vector Machine(SVM) is a supervised machine learning algorithm, that can be used for both classification or regression problems. In the SVM algorithm, we plot each data item as a point, with the value of each feature being the value of the co-ordinate. Then we perform classification by separating groups with a diagonal.

<p align="center">
  <img width="360" height="300" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_1.png">
</p>

## History

The original SVM algorithm was developed by  Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1963. In 1992, Bernhard Boser, Isabelle Guyon, and Vladimir Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes.

## Working

1. Scenario 1 - Identify the right hyper-plane (Scenario-1): Here, we have three hyper-planes (A, B, and C). Now, identify the right hyper-plane to classify stars and circles. For this, you need to remember a thumb rule, “Select the hyperplane which separates the two classes better”. In this scenario, hyper-plane “B” has excellently performed this job.

<p align="center">
  <img width="360" height="300" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_21.png">
</p>

2. Scenario 2 - Identify the hyper-plane which has maximum distance from both the groups. In the below figure C does the best job. The hyper-plane with maximum distance from both the groups is known as Margin.

<p align="center">
  <img width="360" height="300" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_3.png">
</p>


3. Scenario 3 - For the below scenerio we need to distinguish between the groups for that we need to choose A. To seprate stars and circles.

<p align="center">
  <img width="360" height="300" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/10/SVM_5.png">
</p>

## Pro's

- It works really well with a clear margin of separation
- It is effective in high dimensional spaces.
- It is effective in cases where the number of dimensions is greater than the number of samples.
- It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

## Con's

- It doesn’t perform well when we have large data set because the required training time is higher.
- It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping.
- SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is included in the related SVC method of Python scikit-learn library.

## Resources
- [For more details](https://en.wikipedia.org/wiki/Support-vector_machine)
- [SVM source code](https://gist.github.com/mblondel/586753)
- [Examples and implementation of SVM](https://github.com/topics/support-vector-machine)
