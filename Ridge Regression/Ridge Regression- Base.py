#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[1]:


class Ridge_Regression():  #defining a class named Ridge Regression
    
    def _init_(self,iteration,lam,alpha):  #using _init_ method which builds a constructor and initializing parameters
        
        self.iteration = iteration   #number of iterations
        self.lam = lam               #value for lambda 
        self.alpha = alpha           #alpha tuning parameter
    
    def fit(self,x,y):
        
        m = x.shape[0]     #getting the no. of data points
        
        # #initialising weights on the basis of number of input parameters
        
        self.w = np.zeros((x.shape[1],1))
        self.b = 0
        self.x = x
        self.y = y
        
        for _ in range(self.iteration):
           
            Yp =np.dot(x,self.w) + b.self  #calculating the predicted values  
            
            residuals =  self.y-yi    #calculating the residuals
            
            #calculating gradients
            
            
            gradient_w = (-2*np.dot(x.T,residuals)) + 2 * self.w * self.lam /self.m 
            
            gradient_b = - 2 * np.sum( residuals ) / self.m 
            
            #updating weights
            
            self.w = self.w - self.alpha*gradient_w
            self.b = self.b - self.alpha*gradient_b
            
            return self
        
    def predict(self,x):
        
        return np.dot(x,self.w)  + b.self    
    
        
    


# In[ ]:




