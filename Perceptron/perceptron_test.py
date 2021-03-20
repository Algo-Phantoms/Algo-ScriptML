# Perceptron
# Maths behind Perceptron Training

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from perceptron_training import Perceptron

# ------- Generating the dataset using make_blobs -------
X,Y = make_blobs(n_samples=800, centers=2, n_features=2, random_state=2)
plt.style.use("seaborn")
plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Accent)
plt.show()

# -------- Splitting train and test --------- 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.3,random_state = 101)

# -------- Predicting using Perceptron class --------
p = Perceptron()
p.fit(Xtrain, Ytrain)
pred = p.predict(Xtest)

print(p.accuracy(Ytest,pred))
