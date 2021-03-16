import numpy as np
import matplotlib.pyplot as plt

m = 100 #number of training examples
n = 2 #number of features 

X = np.zeros(shape = [m,n])
y = np.zeros(shape = [m,1])

#normalizing the features (using mean and standard deviation)
def normalize_features(X):
    for i in range(1,X.shape[1]+1):
        mean = np.mean(X[:,i-1])
        diff = np.amax(X[:,i-1])-np.amin(X[:,i-1])
        X[:,i-1] = (X[:, i-1]-mean)/diff

#computing the hypothetical function
def hypothetical_function(X, theta):
    ones_array = np.ones(shape=[X.shape[0]])
    X_temp = np.insert(X, 0, ones_array, axis = 1).reshape(X.shape[0],X.shape[1]+1)
    return((1/(1+np.exp(-np.sum(theta.transpose()*X_temp, axis = 1)))).reshape(X.shape[0],1))
    
#cost function
def compute_cost(X, y, theta):
    #for y==1
    J1 = np.sum(y*np.log(hypothetical_function(X, theta)))
    #for y==0
    J2 = np.sum((1-y)*np.log(1-hypothetical_function(X,theta)))
    J = (-1/X.shape[0])*(J1 + J2)
    return J

#gradient descent for cost function optimization
def gradient_descent(X, y, theta, alpha):
    ones_array = np.ones(shape=[X.shape[0]])
    X_temp = np.insert(X, 0, ones_array, axis = 1).reshape(X.shape[0],X.shape[1]+1)
    delta = np.sum((hypothetical_function(X,theta)-y)*X_temp,axis = 0).reshape(theta.shape[0],1)
    theta = theta - (alpha*delta)
    return theta

#implementing the logistic regression for testing
def logistic_regression(X, y, theta, alpha, num_itr):
    cost = []
    for i in range (1,num_itr+1):
        theta = gradient_descent(X, y, theta, alpha)
        cost.append(compute_cost(X, y, theta))
    print(theta)
    plt.plot(cost)
    plt.show()
    return theta

#testing the algorithm
f = open("dataset1.csv", "r")
i = 0
j = 0
for line in f:
    str = ""
    for char in line:
        if (char == '\n' and j == 2):
                y[i] = (float)(str)
                str = ""
                continue
        if(char == ','):
            X[i][j] = (float)(str)
            str=""
            j= j+1
            continue
        str+=char
    i = i+1
    j = 0
#print(X)
#print(y)
normalize_features(X)
plt.plot(X[:,0], X[:,1], 'o')
plt.show()
theta = np.ones(shape=[n+1,1])
print(X.shape)
print(theta.shape)
print(compute_cost(X, y, theta))
theta = logistic_regression(X, y, theta, 0.01, 2500)

x_val = np.linspace(-0.5,0.5,100)
y_val = -(theta[0]+theta[1]*x_val)/theta[2]
plt.plot(X[:,0],X[:,1],'o')
plt.plot(x_val,y_val)
plt.show()
    