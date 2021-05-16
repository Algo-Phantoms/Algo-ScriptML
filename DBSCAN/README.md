## Introduction

Clustering is a method of grouping a series of data points in such a way that they are grouped together based on their similarity. As a consequence, clustering algorithms search for similarities and discrepancies between data points. Clustering is an unsupervised learning process, which means that data points have no labels attached to them. The algorithm seeks to figure out the data's underlying structure.
DBSCAN(Density-Based Spatial Clustering of Applications with Noise) algorithm is a clustering algorithm that can detect arbitrary formed clusters as well as clusters with noise (i.e. outliers).
DBSCAN's underlying concept is that a point belongs to a cluster if it's close to a lot of other points from that cluster. The two main parameters are:

eps(epsilon)- This is defined as the radius of the neighbourhood from a core point(x).

minPts - Minimum points to be present in a points neighbourhood.

![image](https://user-images.githubusercontent.com/67017422/113477225-06842800-949e-11eb-9285-9c2490b5dff9.png)

## Advantages of using DBSCAN algorithm over other clustering algorithm:-

1. Unlike K-means we don't need to provide the model with number of cluster in prior, which was a challenge in K-means. In DBSCAN we just have to provide with the distance theat is considered "close" for creating a cluster.
2. K-Means clustering may bring together unrelated observations. Even if the results are spaced far apart in the vector space, they ultimately become a part of a cluster. Since clusters are formed by the mean value of cluster components, each data point contributes to the formation of clusters. A small shift in data points may have an effect on the clustering result. Because of the way clusters are created in DBSCAN, this issue is greatly reduced. Unless we come across some strange shape info, this is typically not a major issue.

## Classification of points:-

1. Core points- A point is a core point if it is surrounded by at least minPts number of points with radius eps (including the point itself).Hence a point is core point only when no. of neighbours >= minPts.

2. Boundary points- A point is a border point if it can be reached from a central point and the number of points in its immediate vicinity is less than minPts. Hence the necessary condition for a point to be boundary point is no. of neighbours < minPts.

3. Outlier/ Noise- If a point is neither a core point nor a boundary point then it is noise.

![image](https://user-images.githubusercontent.com/67017422/113477997-abedca80-94a3-11eb-9aea-9576567714ba.png)

## Density edge:-

If x and y are core points and the distance between them (x,y)<= eps, then we can join these points. This edge is known as Density edge.

## Density Connected Points:-

If both x and y are core points and a path formed by density edges connects point (x) to point (y), they are said to be density connected points (y).

## Steps For This Algorithm:-

1. Classify the points.
2. Discard Noise.
3. Assign cluster to a core point.
4. Color all the density connected points of a core points.
5. Color boundary points according tot the nearest core point.

For further knowledge, you may also refer to https://www.youtube.com/watch?v=Q7iWANbkFxk
