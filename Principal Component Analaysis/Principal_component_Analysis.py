#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = np.load("X_train.npy")
y = np.load("y_train.npy")

for i in range(10):
    digit = np.mean(X[y == i], axis = 0)
    digit = digit.reshape(28,28)
    plt.subplot(2, 5, i+1)
    plt.imshow(digit, cmap = "gray")

mean = np.mean(X, axis = 0)
mean_img = np.array(mean).reshape(28,28)
plt.imshow(mean_img, cmap='gray')
plt.savefig('elbow.png', bbox_inches='tight')
X_std = (X - mean)
cov_mat = np.cov(X_std, rowvar = 0)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eigen_values_sorted = np.argsort(-eig_vals)
print(np.shape(eig_vals))
plt.ylabel('Eigen Value')
plt.xlabel('# Principal components')
plt.ylim(0, 1.1)
plt.legend(loc = 'best')
plt.scatter(range(eig_vals.shape[0]),eig_vals[eigen_values_sorted])
plt.savefig('elbow.png', bbox_inches='tight')
plt.figure()
plt.tight_layout()

def plot_mnist_pca(X_pca, y):
    markers = 's','x','o','.',',','<','>', '^','8','*'
    colors = list(plt.rcParams['axes.prop_cycle'])
    target = np.unique(y)
    print(list(zip(target,markers)))
    for idx, (t, m) in enumerate(zip(target, markers)):
        subset = X_pca[y == t]
        plt.scatter(subset[:, 0], subset[:, 1], s = 50,c = colors[idx]['color'], label = t, marker = m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc = 'lower left')
    plt.tight_layout()
    plt.savefig('pca_projection.png', bbox_inches='tight')
    plt.figure()
    plt.show()

X_std_pca = X_std.dot(eig_vecs[:,eigen_values_sorted[:2]])
plot_mnist_pca(X_std_pca, y)
print(eig_vecs[:,eigen_values_sorted[:5]].shape)
for i in range(5):
    reshape = eig_vecs[:,eigen_values_sorted[i]].reshape(28,28)
    plt.subplot(2,3,i+1)
    plt.imshow(reshape.real, cmap = "gray")
    plt.savefig(str(i)+'.png', bbox_inches='tight')


# In[ ]:




