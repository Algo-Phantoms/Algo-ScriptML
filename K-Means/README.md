# K-Means

## Introduction

K-Means clustering is one of the simplest popular unsupervised machine learning algorithm. The algorithm discovers K (non-overlapping) clusters by finding K centroids ("central" points) and then assigns each point to the cluster associated with its nearest centroid. 

K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells. 

<p align="center">
  <img width="460" height="300" src="https://media.giphy.com/media/12vVAGkaqHUqCQ/giphy.gif">
</p>

## How K-Means Algorithm Works?

The algorithm starts with an inital set of cluster Centroids chosen at random or according to some heuristic procedure. In each iteration, each data point is assigned to its nearest cluster Centroid. Nearness is measured using the Euclidean distance measure. Then the cluster Centroids are re-computed. The Centroid of each cluster is calculated as the mean value of all the data points that are assigned to that cluster. 

Several termination conditions are possible. For example, the search may stop when the error that 
is computed at every iteration does not reduce because of reassignment of the Centroids. This indicates that the present partition is locally optimal. Other stopping criteria can be used also such as stopping the algorithms after a pre-defined number of iterations.

**Steps for implementation**

[![KMeans.png](https://i.postimg.cc/zD6MQfsX/KMeans.png)](https://postimg.cc/bd03DqYK)

## Advantages
1. The algorithm has a linear complexity, making it computationally attractive.
2. It is also simple to implement and easy to intrepret.
3. It has a high speed of convergence and adaptability to sparse data.

## Disadvantages
1. K-Means involves selection of initial parition or the initial Centroids, making it very sensitive to this selection. This may make the difference between the algorithm converging at global or local minimum.
2. K-Means algorithm only works well on datasets having circular clusters.
3. It is also sensitive to noisy data and outliers, and a single outlier can increase the squared error dramatically.
4. It is applicable only when the mean is defined (namely, for numeric attributes).
5. It also requires the number of clusters in advance, which is not trival when no prior knowledge is available.

## Reference
Big Data Analytics - By Radha Shankarmani and M. Vijayalakshmi