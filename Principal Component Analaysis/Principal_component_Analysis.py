#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Commented out IPython magic to ensure Python compatibility.
#import the required libraries
import pandas as pd   #step-1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

#read the dataset    #step-2
data=pd.read_csv('HR_comma_sep.csv')
data.head()

#get the summary of the data
data.describe()

columns_names=data.columns.tolist()
print("Columns names:")
print(columns_names)

data.shape   

data.head()

data.corr()

correlation=data.corr()  #step-3
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.title('Correlation Between different features')

data['sales'].unique()  #step-4

sales=data.groupby('sales').sum()
sales

data['sales'].unique()

groupby_sales=data.groupby('sales').mean()
groupby_sales

IT=groupby_sales['satisfaction_level'].IT
RandD=groupby_sales['satisfaction_level'].RandD
accounting=groupby_sales['satisfaction_level'].accounting
hr=groupby_sales['satisfaction_level'].hr
management=groupby_sales['satisfaction_level'].management
marketing=groupby_sales['satisfaction_level'].marketing
product_mng=groupby_sales['satisfaction_level'].product_mng
sales=groupby_sales['satisfaction_level'].sales
support=groupby_sales['satisfaction_level'].support
technical=groupby_sales['satisfaction_level'].technical
technical

#step-5
department_name=('sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD')
department=(sales, accounting, hr, technical, support, management,
       IT, product_mng, marketing, RandD)
y_pos = np.arange(len(department))
x=np.arange(0,1,0.1)

plt.barh(y_pos, department, align='center', alpha=0.8)
plt.yticks(y_pos,department_name )
plt.xlabel('Satisfaction level')
plt.title('Mean Satisfaction Level of each department')

#step-6
data.head()
data_drop=data.drop(labels=['sales','salary'],axis=1)
data_drop.head()
cols=data_drop.columns.tolist()
cols
cols.insert(0,cols.pop(cols.index('left')))
cols
data_drop = data_drop.reindex(columns= cols)

#step-7

X = data_drop.iloc[:,1:8].values
y = data_drop.iloc[:,0].values
X
y

np.shape(X)
np.shape(y)

#step-8
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

plt.figure(figsize=(8,8))
sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different features')

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#step-9
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(7), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
  
 #step-10
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1), 
                      eig_pairs[1][1].reshape(7,1)
                    ))
print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)
Y

#step-11
from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

"""The above plot shows almost 90% variance by the first 6 components. Therfore we can drop 7th component."""

from sklearn.decomposition import PCA 
sklearn_pca = PCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print(Y_sklearn)

Y_sklearn.shape

