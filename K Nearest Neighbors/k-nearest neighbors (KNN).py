# %% [code]
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


## For testing Purposes
'''
## Importing libraries

import sklearn.datasets
import matplotlib.pyplot as plt

## creating dataset

x,y = sklearn.datasets.make_classification(n_samples=1000, n_classes=2,
n_clusters_per_class=1, n_features=2,n_informative=2, n_redundant=0, n_repeated=0)


## Visualization

query_p = np.array([0.5,0.5])   
plt.scatter(query_p[0],query_p[1],c = 'r') ## plot the query point
plt.scatter(x[:,0],x[:,1],c = y)
plt.show()


## testing the algorithm

result = knn(x,y,query_p)    ### query point ==> x = 0.5,y = 0.5  
print(result)
'''