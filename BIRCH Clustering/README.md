# BIRCH CLUSTERING

## Introduction

Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) is a clustering algorithm that can cluster large datasets by first generating a small and compact summary of the the large dataset that retains as much information as possible. This smaller summary is then clustered instead of clustering the larger dataset.

## Need for BIRCH

Clustering algorithms like K-means clustering do not perform clustering very efficiently and it is difficult to process large datasets with a limited amount of resources (like memory or a slower CPU). So, regular clustering algorithms do not scale well in terms of running time and quality as the size of the dataset increases. This is where BIRCH clustering is found to be very useful.

## CF Tree

BIRCH is based on the notation of CF (Clustering Feature); a CF Tree. It is a height balanced tree that stores the clustering features for a hierarchical clustering. A CF Tree structure is given as below:

1. Each non-leaf node has at most B entries.

2. Each leaf node has at most L CF entries which satisfy threshold T, a maximum diameter of radius.

3. P(page size in bytes) is the maximum size of a node.

4. Compact: each leaf node is a subcluster, not a data point.

![](https://i.imgur.com/mkSo8wI.png)

## Advantages

▪ Finds a good clustering with a single scan and improves the quality with a few additional scans.

▪ Works with very large data sets.

## Disadvantages

▪ Handles only numeric data.

## Applications

▪ Pixel classification in images.

▪ Image compression.

## References

▪ https://www.geeksforgeeks.org/ml-birch-clustering/

▪ https://www.ques10.com/p/9298/explain-birch-algorithm-with-example/
