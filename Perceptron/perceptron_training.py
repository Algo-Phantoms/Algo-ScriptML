# Perceptron
# Maths behind Perceptron Training

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ------- Generating the dataset using make_blobs -------
X,Y = make_blobs(n_samples=800, centers=2, n_features=2, random_state=2)
print(X.shape, Y.shape)

plt.style.use("seaborn")
plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Accent)
plt.show()

# -------- MODEL AND HELPER FUNCTIONS ---------

# Sigmoid function is an activation function (denoted as sigma(z)). The output of the sigma(z) belongs to the range 0 to 1. 
# 0 means - highly negative input and 1 means - highly positive input 
# This is useful as an activation function when one is interested in probability mapping rather than precise values of input parameter t.

def sigmoid(z):
    return (1.0)/(1+np.exp(-z))

# -------- IMPLEMENT PERCEPTRON LEARNING ALGORITHMS -------
# - Learn weights
# - Reduce the loss
# - Make the predictions

def predict(X,weights):
    """ X -> mx(n+1) matrix, w -> (nx1)matrix """
    z = np.dot(X,weights)
    predictions = sigmoid(z)
    return predictions
def loss(X,Y,weights):
    """ Binary Cross Entropy """
    Y_ = predict(X,weights)
    cost = np.mean(-Y*np.log(Y_) - (1-Y)*np.log(1-Y_))
    return cost
def update(X,Y,weights,lr):
    """ Perform weight updates for 1 epoch """
    Y_ = predict(X,weights)
    dw = np.dot(X.T,Y_-Y)
    m = X.shape[0]
    weights = weights - lr*dw/(float(m))
    return weights
def train(X,Y,lr=0.3,maxEpochs = 1000):
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones,X))
    weights = np.zeros(X.shape[1])
    for epoch in range(maxEpochs):
        weights = update(X,Y,weights,lr)
        if epoch%10==0:
            l=loss(X,Y,weights)
            print("Epoch %d Loss %.4f"%(epoch,l))
    return weights

# - We can not use Mean Squared Error loss fucntion because it is a non convex function. When we minimize
# the loss using gradient descent there will be high probabilty of getting stuck at local minima.
# - So to overcome this difficulty we will use Log Loss

weights = train(X,Y)

# -------- Activation Function ---------
# 1. The activation function to be used is a subjective decision taken by the data scientist, 
# based on the problem statement and the form of the desired results.
# 2. If the learning process is slow or has vanishing or exploding gradients, 
# the data scientist may try to change the activation function to see if these problems can be resolved.

# ------- PERCEPTRON IMPLEMENTATION PART II -------
# - MAKE PREDICTION
# - VISUALISE DESCISION SURFACE 
# - LINEAR VS NON LINEAR CLASSIFICATION

def getpredict(X_test,weights,labels = True):
    if X_test.shape[1]!=weights.shape[0]:
        ones = np.ones((X_test.shape[0],1))
        X_test = np.hstack((ones,X_test))
        
    probs = predict(X_test,weights)
    if not labels:
        return probs
    else:
        labels = np.zeros(probs.shape)
        labels[probs>=0.5] = 1
        return labels

print(weights)

# Data Visualization
x1 = np.linspace(-10,2,10)
print(x1)
x2 = -(weights[0]+weights[1]*x1)/weights[2]
print(x2)

plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Accent)
plt.plot(x1,x2,c='red')
plt.show()

# Accuracy
y_ = getpredict(X,weights,labels= True)
training_acc = np.sum(y_==Y)/Y.shape[0]
print(training_acc)