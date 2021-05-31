# MINI BATCH K-MEANS CLUSTERING

## Introduction

Mini Batch K-means algorithm‘s main idea is to use small random batches of data of a fixed size, so they can be stored in memory. Each iteration a new random sample from the dataset is obtained and used to update the clusters and this is repeated until convergence. Each mini batch updates the clusters using a convex combination of the values of the prototypes and the data, applying a learning rate that decreases with the number of iterations. This learning rate is the inverse of the number of data assigned to a cluster during the process. As the number of iterations increases, the effect of new data is reduced, so convergence can be detected when no changes in the clusters occur in several consecutive iterations.

## Need for Mini-Batch K-Means Clustering Algorithm

K-means is one of the most popular clustering algorithms, mainly because of its good time performance. With the increasing size of the datasets being analyzed, the computation time of K-means increases because of its constraint of needing the whole dataset in main memory. For this reason, several methods have been proposed to reduce the temporal and spatial cost of the algorithm. A different approach is the Mini batch K-means algorithm.

The empirical results suggest that it can obtain a substantial saving of computational time at the expense of some loss of cluster quality, but not extensive study of the algorithm has been done to measure how the characteristics of the datasets, such as the number of clusters or its size, affect the partition quality.

<img src = 'https://media.geeksforgeeks.org/wp-content/uploads/20190510082812/index16.png' width = 500, height = 250 />

## Advantages

▪ If variables are huge, then K-Means most of the times gets computationally faster than hierarchical clustering, if we keep k small.

▪ K-Means produce tighter clusters than hierarchical clustering.

## Disadvantages

▪ Difficult to predict K-Value.

▪ It may not work well with clusters (in the original data) of different size and different density.

## References

▪ https://www.geeksforgeeks.org/ml-mini-batch-k-means-clustering-algorithm/

▪ http://playwidtech.blogspot.com/2013/02/k-means-clustering-advantages-and.html
