import numpy as np 

class Stochastic_gradient_descent():

    def __init__(self,learning_rate=0.1):
        self.learning_rate=learning_rate
    
    def fit(self,X,Y):
        '''
        first finding the size of given features.

        m=no. of given training sets
        n=no.features in each training set

        after that we will assign the theta to 
        the random values.

        '''
        m,n=X.shape
        self.m,self.n=X.shape

        # now initializing the theta.

        self.theta=np.random.random((self.n+1,1))
        self.X=X
        self.Y=Y
        '''
        we are adding a column vector having value 1,because
        the hypothesis function for stochastic gradient descent
        algo is:h(theta)=theta(0)+theta(1)*X(1)+theta(2)*X(2)...........
        so, In order to vectorize the hypothesis fn we will add column vector.
        Here is breif explanation: X.shape=(m,n) and theta.shape=(n+1,1)
        so, in order to multiply the both matrix, we have to add one column to
        X, so that X will become: (m,n+1)
        
        '''
        ones=np.ones(self.m,int).reshape(-1,1)
        self.X=np.concatenate((ones,self.X),axis=1)
        self.gradient_descent()
        return self
    
    def gradient_descent(self):
            
        #updating theta.
        for i in range(self.m):
            grad_cost=self.X[i,:]@self.theta-self.Y[i,:]
            self.theta=self.theta-(self.learning_rate*grad_cost*(self.X[i,:].reshape(-1,1)))
        return self

    
    def predict(self,X):
        m,n=X.shape
        ones=np.ones(m,int).reshape(-1,1)
        X=np.concatenate((ones,X),axis=1)
        return (X@self.theta)
