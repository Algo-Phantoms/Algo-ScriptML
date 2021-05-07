#!/usr/bin/env python
# coding: utf-8

# ### KFold cross validation

# In[40]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf


# In[10]:


for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import pandas as pd


# In[39]:


from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
 
 
data = load_breast_cancer(as_frame = True)
df = data.frame
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
 
k = 5
kf = KFold(n_splits=k, random_state=None)
model = LogisticRegression(solver= 'liblinear')
 
result = cross_val_score(model , X, y, cv = kf)
 
print("Avg accuracy: {}".format(result.mean()))


# In[24]:


df = pd.DataFrame({'y': [6, 8, 12, 14, 14, 15, 17, 22, 24, 23],
                   'x1': [2, 5, 4, 3, 4, 6, 7, 5, 8, 9],
                   'x2': [14, 12, 12, 13, 7, 8, 7, 4, 6, 5]})


# ### Leave-One-Out

# In[38]:


#define predictor and response variables
X = df[['x1', 'x2']]
y = df['y']

# define cross-validation method to use
cv = LeaveOneOut()

# build multiple linear regression model
model = LinearRegression()

# use LOOCV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

# view mean absolute error
mean(absolute(scores))


# ### Leave-P-Out

# In[37]:


# define predictor and response variables
X = df[['x1', 'x2']]
y = df['y']

#define cross-validation method to use
cv = LeavePOut(2)

#build multiple linear regression model
model = LinearRegression()

#use LeavePOut to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

#view RMSE
sqrt(mean(absolute(scores)))

3.619456476385567


# ### K StratifiedFold

# In[41]:


from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets


cancer = datasets.load_breast_cancer()
# Input_x_Features.
x = cancer.data                         
  
# Input_ y_Target_Variable.
y = cancer.target                       
    
   
# Feature Scaling for input features.
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
   
# Create  classifier object.
lr = linear_model.LogisticRegression()
   
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []
   
for train_index, test_index in skf.split(x, y):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold))
   
# Prining Min, Max adnd Overall accuracy
print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified)*100, '%')

print('\nMinimum Accuracy:',
      min(lst_accu_stratified)*100, '%')

print('\nOverall Accuracy:',
      mean(lst_accu_stratified)*100, '%')

print('\nStandard Deviation is:', stdev(lst_accu_stratified))

