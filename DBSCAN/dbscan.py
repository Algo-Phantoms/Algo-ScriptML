#importing libraries
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
class cluster:

    x,_= make_blobs(n_samples=500,n_features=2,centers=4,random_state=19)
    eps=4
    minpts=5
    D=x

    def update_labels(x,pt,eps,labels,cluster_val):
        neighbors=[]
        label_index=[]
        for i in range (0,x.shape[0]):
        
            if np.linalg.norm(x[pt]-x[i])<eps:
                neighbors.append(x[i])
                label_index.append(i)
        if len(neighbors)<minpts:
            for j in range (len(labels)):
                if i in label_index:
                    labels[j]=-1
        else:
            for j in range (len(labels)):
                if i in label_index:
                    labels[j]=cluster_val
        return labels

    labels=[0]*len(D)
    c=1
    for p in range(0,D.shape[0]):
        if labels[p]==0:
            labels=update_labels(D,p,eps,labels,c)
            c=c+1
        
    def plotRes(data, clusterRes, clusterNum):
        nPoints = len(data)
        scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
        for i in range(clusterNum):
            if (i==0):
                #Plot all noise point as blue
                color='blue'
            else:
                color = scatterColors[i % len(scatterColors)]
            x1 = [];  y1 = []
            for j in range(nPoints):
                if clusterRes[j] == i:
                    x1.append(data[j, 0])
                    y1.append(data[j, 1])
            plt.scatter(x1, y1, c=color, alpha=1, marker='.')
        
    plotRes(x,labels,c)
    plt.show()
