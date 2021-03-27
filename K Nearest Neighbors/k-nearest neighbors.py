#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2)) # calculating distance

# main algo 
def knn(X,Y,queryPoint,k=5):
    
    vals = [] # creating list to append all distances
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i])) #appending all distances 
        
    #sorting the list
    vals = sorted(vals)
    # choose first k distances 
    vals = vals[:k]
    
    vals = np.array(vals)

    
    new_vals = np.unique(vals[:,1],return_counts=True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


# In[ ]:




