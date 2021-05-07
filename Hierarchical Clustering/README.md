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


## Advantages

1) No apriori information about the number of clusters required.
2) Easy to implement and gives best result in some cases.

## Disadvantages

1) Algorithm can never undo what was done previously.
2) Time complexity of at least O(n2 log n) is required, where ‘n’ is the number of data points.
3) Based on the type of distance matrix chosen for merging different algorithms can suffer with one or more of the following:
    i) Sensitivity to noise and outliers
    ii) Breaking large clusters
    iii) Difficulty handling different sized clusters and convex shapes
4) No objective function is directly minimized
5) Sometimes it is difficult to identify the correct number of clusters by the dendogram.


## Reference
• https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/

• https://github.com/AkhileshThite/Clustering-Mall-Customers

• https://towardsdatascience.com/hierarchical-clustering-explained-e58d2f936323

