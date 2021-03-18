import numpy as np

class LassoRegression() : 
    
    def __init__( self, learning_rate, epoch,lamda ) : 
        '''
        lamda : L1_penality
        '''
        self.learning_rate = learning_rate 
        self.epoch = epoch 
        self.lamda = lamda 
        
       
    def fit( self, X, Y ) : 
        '''
        X.shape = (m,n)
        Y.shape = (m,)
        
        where,
            m = number of training examples
            n = number of features 
        '''
        
        self.m, self.n = X.shape
        # Initializing weights
        
        self.a = np.zeros( self.n ) 
        self.b = 0
        self.X = X 
        self.Y = Y 

        # calculating gradient descent epooch time
                
        for i in range( self.epoch ) : 
            self.updating_weights()
            
        return self
        
    def updating_weights( self ) :
        '''
        updating weights using gradient descent 
        '''        
            
        Y_pred = self.predict( self.X ) 
        
        # calculating gradients 
        da = (-2*(self.X.T).dot(self.Y-Y_pred)+self.lamda*np.sign(self.a))/self.m
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
        
        # updating weights 
        self.a  -= self.learning_rate * (da) 
        self.b  -= self.learning_rate * (db) 
        
        return self
    

    def predict( self, X ) : 
        '''
        X.shape = (m,n)
        
        where,
            m = number of training examples
            n = number of features 
            
        output : (m,1)
        '''
        return X.dot( self.a ) + self.b 
    

