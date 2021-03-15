# -*- coding: utf-8 -*-
import random
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Library Implementation of K Means


X, y = make_blobs(centers=3, random_state=42)


sns.scatterplot(X[:, 0], X[:, 1], hue=y)

sns.scatterplot(X[:, 0], X[:, 1])


model = KMeans(n_clusters=4)

model.fit(X)

y_gen = model.labels_

sns.scatterplot(X[:, 0], X[:, 1], hue=y_gen)

model.cluster_centers_

sns.scatterplot(X[:, 0], X[:, 1], hue=y_gen)

for center in model.cluster_centers_:
    plt.scatter(center[0], center[1], s=60)

sns.scatterplot(X[:, 0], X[:, 1])


# Custom Implementation of K Means

class Cluster:

    def __init__(self, center):
        """Initialization of Clusters for K Means."""
        self.center = center
        self.points = []

    def distance(self, point):
        return np.sqrt(np.sum((point - self.center) ** 2))


class CustomKMeans:

    def __init__(self, n_clusters=3, max_iters=20):
        """Custom Implementation of K Means."""
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):

        clusters = []
        for _ in range(self.n_clusters):
            cluster = Cluster(center=random.choice(X))
            clusters.append(cluster)

        for _ in range(self.max_iters):

            labels = []

            # going for each point
            for point in X:

                # collecting disctances form every cluster
                distances = []
                for cluster in clusters:
                    distances.append(cluster.distance(point))

                # finding closest cluster
                closest_idx = np.argmin(distances)
                closest_cluster = clusters[closest_idx]
                closest_cluster.points.append(point)
                labels.append(closest_idx)

            for cluster in clusters:
                cluster.center = np.mean(cluster.points, axis=0)

        self.labels_ = labels
        self.cluster_centers_ = [cluster.center for cluster in clusters]


model = CustomKMeans(n_clusters=2)

model.fit(X)

sns.scatterplot(X[:, 0], X[:, 1], hue=model.labels_)

for center in model.cluster_centers_:
    plt.scatter(center[0], center[1], s=60)
