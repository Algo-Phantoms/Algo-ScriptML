import numpy as np

#normalizing the features
def normalize_features(X):
    for i in range(1,X.shape[1]+1):
        mean = np.mean(X[:,i-1])
        rng = np.amax(X[:,i-1])-np.amin(X[:,i-1])
        X[:,i-1] = (X[:, i-1]-mean)/rng

#computing the hypothetical function
def hypothetical_function(X, theta):
    ones_array = np.ones(shape=[X.shape[0]])
    #adding intercept term
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
    #calculating the gradient
    delta = np.sum((hypothetical_function(X,theta)-y)*X_temp,axis = 0).reshape(theta.shape[0],1)
    theta = theta - (alpha*delta)#updating theta
    return theta

#training the model
def logistic_regression(X, y, theta, alpha, num_itr):
    normalize_features(X)
    for i in range (1,num_itr+1):
        theta = gradient_descent(X, y, theta, alpha)
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

