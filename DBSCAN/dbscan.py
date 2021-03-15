# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:37:47 2021

@author: Lenovo
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# calculating homogeneity score, silhoutte score, completeness score and measure 
# score and various other information of the cluster
h_score= metrics.homogeneity_score(labels_true, labels)
c_score= metrics.completeness_score(labels_true, labels)
m_score= metrics.v_measure_score(labels_true, labels)
adjusted_score= metrics.adjusted_rand_score(labels_true, labels)
mutual_score= metrics.adjusted_mutual_info_score(labels_true, labels)
silhouette_coeff= metrics.silhouette_score(X, labels)
print("Homogeneity: %0.3f" % h_score)
print("Completeness: %0.3f" % c_score)
print("V-measure: %0.3f" % m_score)
print("Adjusted Rand Index: %0.3f" % adjusted_score)
print("Adjusted Mutual Information: %0.3f" % mutual_score )
print("Silhouette Coefficient: %0.3f" % silhouette_coeff)

# Plot result
import matplotlib.pyplot as plt

# Black is removed.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Number of clusters estimated are: %d' % n_clusters_)
plt.show()
