#importing libraries

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame

#using sklearn
from sklearn.cluster import DBSCAN

class cluter:
    x,_= make_blobs(n_samples=500,n_features=2,centers=4,random_state=19)
    clustering= DBSCAN(eps=4 , min_samples=5).fit(x)
    cluster=clustering.labels_
    print(len(set(cluster)))
    
    def show_cluster(x,cluster):
        df=DataFrame(dict(x=x[:,0],y=x[:,1],label=cluster))
        colors={-1:'red',0:'blue',1:'orange',2:'skyblue'}
        grouped=df.groupby('label')
        fig,ax=plt.subplots(figsize=(8,5))
        for key,group in grouped:
            group.plot(ax=ax,kind='scatter',x='x',y='y',label=key,color=colors[key])
        plt.show()

    show_cluster(x,cluster)
