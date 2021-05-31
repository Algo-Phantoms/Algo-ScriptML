# XGBoost

## Introduction
It stands for “Extreme Gradient Boosting”, where the term “Gradient Boosting” originates from the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman.

## What is XGBoost?
XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data. XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.). A wide range of applications: Can be used to solve regression, classification, ranking, and user-defined prediction problems. It is a perfect blend of software and hardware capabilities designed to enhance existing boosting techniques with accuracy in the shortest amount of time.

**Evolution of XGBoost Algorithm from decision Trees**

![Image1](https://miro.medium.com/max/1850/1*QJZ6W-Pck_W7RlIDwUIN9Q.jpeg)

## XGBoost Features:
- **Regularized Learning:** Regularization term helps to smooth the final learnt weights to avoid over-fitting. The regularized objective will tend to select a model employing simple and predictive functions.
- **Gradient Tree Boosting:** The tree ensemble model cannot be optimized using traditional optimization methods in Euclidean space. Instead, the model is trained in an additive manner.
- **Shrinkage and Column Subsampling:** Besides the regularized objective, two additional techniques are used to further prevent overfitting. The first technique is shrinkage introduced by Friedman. Shrinkage scales newly added weights by a factor η after each step of tree boosting. Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each tree and leaves space for future trees to improve the model.

## Splitting Algorithms:
- **Exact Greedy Algorithm:** The main problem in tree learning is to find the best split. This algorithm enumerates over all the possible splits on all the features. It is computationally demanding to enumerate all the possible splits for continuous features.
- **Approximate Algorithm:** The exact greedy algorithm is very powerful since it enumerates over all possible splitting points greedily. However, it is impossible to efficiently do so when the data does not fit entirely into memory. Approximate Algorithm proposes candidate splitting points according to percentiles of feature distribution. The algorithm then maps the continuous features into buckets split by these candidate points, aggregates the statistics and finds the best solution among proposals based on the aggregated statistics.
- **Weighted Quantile Sketch:** One important step in the approximate algorithm is to propose candidate split points. XGBoost has a distributed weighted quantile sketch algorithm to effectively handle weighted data.
- **Sparsity-aware Split Finding:** In many real-world problems, it is quite common for the input x to be sparse. There are multiple possible causes for sparsity:
<ol type="1">
  <li> Presence of missing values in the data. </li>
  <li> Frequent zero entries in the statistics. </li>
  <li> Artifacts of feature engineering such as one-hot encoding. </li>
</ol>
It is important to make the algorithm aware of the sparsity pattern in the data. XGBoost handles all sparsity patterns in a unified way.

## XGBoost objective function:
The objective function (loss function and regularization) at iteration t that we need to minimize is the following:

![Image2](https://miro.medium.com/max/875/1*cU3rKmPvGZa3gzAZ3tzKnQ.png)

## System Features:
- Parallelization of tree construction using all of your CPU cores during training. Collecting statistics for each column can be parallelized, giving us a parallel algorithm for split finding.
- Cache-aware Access: XGBoost has been designed to make optimal use of hardware. This is done by allocating internal buffers in each thread, where the gradient statistics can be stored.
- Blocks for Out-of-core Computation for very large datasets that don’t fit into memory.
- Distributed Computing for training very large models using a cluster of machines.
- Column Block for Parallel Learning. The most time-consuming part of tree learning is to get the data into sorted order. In order to reduce the cost of sorting, the data is stored in the column blocks in sorted order in compressed format.

## Why does XGBoost Performs so well?
XGBoost and Gradient Boosting Machines (GBMs) are both ensemble tree methods that apply the principle of boosting weak learners (CARTs generally) using the gradient descent architecture. However, XGBoost improves upon the base GBM framework through systems optimization and algorithmic enhancements.

![Image3](https://miro.medium.com/max/1554/1*FLshv-wVDfu-i54OqvZdHg.png)

## Goals of XGBoost:
- Execution Speed: XGBoost was almost always faster than the other benchmarked implementations from R, Python Spark and H2O and it is really faster when compared to the other algorithms.
- Model Performance: XGBoost dominates structured or tabular datasets on classification and regression predictive modelling problems.

![Image4](https://miro.medium.com/max/5248/1*1kjLMDQMufaQoS-nNJfg1Q.png)

## Advantages:
- It is Highly Flexible.
- It uses the power of parallel processing.
- It is faster than Gradient Boosting.
- It supports regularization.
- It is designed to handle missing data with its in-build features.
- The user can run a cross-validation after each iteration.

## Disadvantage:
- The high flexibility results in many parameters that interact and influence heavily the behavior of the approach (number of iterations, tree depth, regularization parameters, etc.). This requires a large grid search during tuning.
- Less interpretative in nature, although this is easily addressed with various tools.
- Computationally expensive - often require many trees (>1000) which can be time and memory exhaustive.

## Conclusion:
XGBoost is a faster algorithm when compared to other algorithms because of its parallel and distributed computing. XGBoost is developed with both deep considerations in terms of systems optimization and principles in machine learning. The goal of this library is to push the extreme of the computation limits of machines to provide a scalable, portable and accurate library.

## References:
- https://xgboost.readthedocs.io/en/latest/tutorials/model.html
- https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d
- https://medium.com/sfu-cspmp/xgboost-a-deep-dive-into-boosting-f06c9c41349
- https://www.mygreatlearning.com/blog/xgboost-algorithm/
- https://blog.paperspace.com/gradient-boosting-for-classification/
