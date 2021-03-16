import numpy as np
import matplotlib.pyplot as plt


#normalizing the features
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

#training the model
def logistic_regression(X, y, theta, alpha, num_itr):
    cost = []
    for i in range (1,num_itr+1):
        theta = gradient_descent(X, y, theta, alpha)
        cost.append(compute_cost(X, y, theta))
    print(theta)
    plt.plot(cost)
    plt.show()
    return theta

#testing on trained model
def calculate_y_predicted(X, y_actual, theta):
    y_predicted = hypothetical_function(X, theta)
    s = 0;
    for i in range(1, len(y_actual)+1):
        if(y_predicted[i-1] > 0.5):
            y_predicted[i-1] = 1
        else:
            y_predicted[i-1] = 0
        if (y_actual[i-1] == y_predicted[i-1]):
            s = s + 1
        print(y_actual[i-1], " ", y_predicted[i-1])
    percent_accuracy = (s/(len(y_actual)))*100
    print( "Accuracy is:", percent_accuracy)


"""#testing the algorithm
m = 100 #number of training examples
n = 2 #number of features

X = np.zeros(shape = [m,n])
y = np.zeros(shape = [m,1])

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
normalize_features(X)
plt.plot(X[:,0], X[:,1], 'o')
plt.show()
theta = np.ones(shape=[n+1,1])
print(X.shape)
print(theta.shape)
print(compute_cost(X, y, theta))
theta = logistic_regression(X[0:60,:], y[0:60], theta, 0.01, 2600)

x_val = np.linspace(-0.5,0.5,100)
y_val = -(theta[0]+theta[1]*x_val)/theta[2]
plt.plot(X[0:60:,0],X[0:60:,1],'o')
plt.plot(x_val,y_val)
plt.show()
calculate_y_predicted(X[60:101,:],y[60:101], theta)"""
