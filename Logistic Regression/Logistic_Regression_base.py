"""
Parameters passed to the functions :

X - array containing all the features

y - array containing the classification values

theta - row vector contaning weights

alpha - learning rate(default = 0.01)

num_itr - number of iterations (default = 100)"""


import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, X, y, theta, alpha=0.01, num_itr = 100):
        self.X = X
        self.y = y
        self.theta = theta
        self.alpha = alpha
        self.num_itr = num_itr

    def debug(self):
        print(self.X.shape[0])
        print(self.y)
        print(self.theta)

    #normalizing the features (using mean and standard deviation)
    def normalize_features(self):
        for i in range(1,self.X.shape[1]+1):
            mean = np.mean(self.X[:,i-1])
            std_dev = np.std(self.X[:,i-1])
            self.X[:,i-1] = (self.X[:, i-1]-mean)/std_dev
        return mean, std_dev

    #computing the hypothetical function
    def hypothetical_function(self):
        ones_array = np.ones(shape=[self.X.shape[0]])
        X_temp = np.insert(self.X, 0, ones_array, axis = 1).reshape(self.X.shape[0],self.X.shape[1]+1)
        return((1/(1+np.exp(-np.sum(self.theta.transpose()*X_temp, axis = 1)))).reshape(self.X.shape[0],1))

    #cost function
    def compute_cost(self):
        #for y==1
        J1 = np.sum(self.y*np.log(self.hypothetical_function()))
        #for y==0
        J2 = np.sum((1-self.y)*np.log(1-self.hypothetical_function()))
        J = (-1/self.X.shape[0])*(J1 + J2)
        return J

    #gradient descent for cost function optimization
    def gradient_descent(self):
        ones_array = np.ones(shape=[self.X.shape[0]])
        X_temp = np.insert(self.X, 0, ones_array, axis = 1).reshape(self.X.shape[0],self.X.shape[1]+1)
        delta = np.sum((self.hypothetical_function()-self.y)*X_temp,axis = 0).reshape(self.theta.shape[0],1)
        self.theta = self.theta - (self.alpha*delta)
        return self.theta

    #training the model
    def logistic_regression(self):
        cost = []
        for i in range (1,self.num_itr+1):
            self.theta = self.gradient_descent()
            cost.append(self.compute_cost())
        plt.plot(cost)
        plt.show()
        return self.theta

    #testing on trained model
    def calculate_y_predicted(self, avg, std_dev):
        for i in range(1,self.X.shape[1]+1):
            self.X[:,i-1] = (self.X[:, i-1]-avg)/std_dev
        y_predicted = self.hypothetical_function()
        s = 0;
        for i in range(1, len(self.y)+1):
            if(y_predicted[i-1] >= 0.5):
                y_predicted[i-1] = 1
            else:
                y_predicted[i-1] = 0
            if (self.y[i-1] == y_predicted[i-1]):
                s = s + 1
            #print(self.y[i-1], " ", y_predicted[i-1])
        percent_accuracy = (s/(len(self.y)))*100
        print( "Accuracy is:", percent_accuracy)

"""
Using the module

import LogisticRegression
log_reg = LogisticRegression(X, y, theta, alpha, num_itr)
mean, standard_deviation = log_reg.normalize_features()(store the mean and deviation to use for prediction and also normalize the data)
log_reg.logistic_regression()(train the model)

For predicting
predict = LogisticRegression(X_test, y_test, theta)
predict.calculate_y_predicted(mean, standard_deviation)

"""


"""
#testing the algorithm
m = 100 #number of training examples
n = 2 #number of features

X = np.zeros(shape = [m,n])
y = np.zeros(shape = [m,1])

link of dataset used ->https://github.com/nikhilkumarsingh/Machine-Learning-Samples/blob/master/Logistic_Regression/dataset1.csv

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
theta = np.zeros(shape=[n+1,1])

x = X.copy()
y_a = y.copy()
graph = LogisticRegression(x ,y_a, theta, 0.01,2600)
graph.normalize_features()

x_ones = []
x_ones2 = []
x_zeros =[]
x_zeros2 =[]
for i in range(1, len(y_a)+1):
    if(y_a[i-1] == 1):
        temp = []
        temp.append(x[i-1][0])
        x_ones.append(temp)
        temp = []
        temp.append(x[i-1][1])
        x_ones2.append(temp)
    else:
        temp = []
        temp.append(x[i-1][0])
        x_zeros.append(temp)
        temp = []
        temp.append(x[i-1][1])
        x_zeros2.append(temp)
plt.plot(x_ones,x_ones2, 'o', color = 'yellow')
plt.plot(x_zeros,x_zeros2,'o', color = 'red')
plt.show()

x_train = X.copy()
y_train = y.copy()
l = LogisticRegression(x_train, y_train,theta,0.01,2600)
m, dev = l.normalize_features()
theta = l.logistic_regression()
x_val = np.linspace(-2,2,100)
y_val = -(theta[0]+theta[1]*x_val)/theta[2]
plt.plot(x_ones,x_ones2, 'o', color = 'yellow')
plt.plot(x_zeros,x_zeros2,'o', color = 'red')
plt.plot(x_val,y_val)
plt.show()

print(theta)
x_test = X[61:101,:].copy()
y_test = y[61:101,:].copy()
p = LogisticRegression(x_test, y_test, theta)
p.calculate_y_predicted(m,dev)
"""
