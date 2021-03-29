# Hierarchical Clustering
## Introduction
Hierarchical clustering is unsupervised machine learning algorithm, as the name suggests is an algorithm that builds hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left.
The results of hierarchical clustering can be shown using dendrogram.

## How to count the number of optimal clusters by using dendrogram?
![](https://github.com/AkhileshThite/Clustering-Mall-Customers/blob/master/Dendrogram.jpeg)

### 3-Steps
1. Start from horizontal line.
2. Select the largest vertical distance.
3. Cross the line over it and you'll get the optimal number of clusters (just like shown in the image).

Note: It's easy to find the optimal no of clusters in K-Means because of elbow method. (we can't find the optimal no of clusters accuratley by using dendogram for large number of data. hence, for large number of data Hierarchical clustering is not good)

## Difference between K Means and Hierarchical clustering

1. Hierarchical clustering can’t handle big data well but K Means clustering can. This is because the time complexity of K Means is linear i.e. O(n) while that of hierarchical clustering is quadratic i.e. O(n2).
2. In K Means clustering, since we start with random choice of clusters, the results produced by running the algorithm multiple times might differ. While results are reproducible in Hierarchical clustering.
3. K Means is found to work well when the shape of the clusters is hyper spherical (like circle in 2D, sphere in 3D).
4. K Means clustering requires prior knowledge of K i.e. no. of clusters you want to divide your data into. But, you can stop at whatever number of clusters you find appropriate in hierarchical clustering by interpreting the dendrogram

## Reference
• https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/

• https://github.com/AkhileshThite/Clustering-Mall-Customers

• https://towardsdatascience.com/hierarchical-clustering-explained-e58d2f936323

