#XGBoost using Numpy (from scratch)

#  The following code is a simple XGBoost model developed using numpy.
# Tha main purpose of this code is to unveil the maths behind XGBoost.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#Consider the following data where the years of experience is predictor variable and salary is the target.

year = [5,7,12,23,25,28,29,34,35,40]
salary = [82,80,103,118,172,127,204,189,99,166]


#  Using regression trees as base learners, we can create a model to predict the salary. 
# For the sake of simplicity, we can choose square loss as our loss function and our objective would be to minimize the square error.
#  As the first step, the model should be initialized with a function F0(x). F0(x) should be a function which minimizes the loss function or MSE (mean squared error)
#  For MSE the Function F minimizes at mean
#  If we had taken MAE , the function would have minimized at median

df = pd.DataFrame(columns=['Years','Salary'])
df.Years = year
df.Salary = salary
df.head()
plt.scatter(x=df.Years,y=df.Salary)


#  The residual is the difference between y and f0 i.e. (y-f0)
#  We can use the residuals from F0(x) to create h1(x). h1(x) will be a regression tree which will try and reduce the residuals from the previous step. The output of h1(x) won’t be a prediction of y; instead, it will help in predicting the successive function F1(x) which will bring down the residuals.

df1 = df


# The additive model h1(x) computes the mean of the residuals (y – F0) at each leaf of the tree. 

# A split is done and the mean of upper part and lower part is calculated 
# Here , I have selected a random split point



for i in range(2):
    f = df.Salary.mean()
    if(i>0):
        df['f'+str(i)] = df['f'+str(i-1)] + df['h'+str(i)]
    else:
        df['f'+str(i)] = f
    df['y-f'+str(i)] = df.Salary - df['f'+str(i)]
    splitIndex = np.random.randint(0,df.shape[0]-1)
    a= []
    h_upper = df['y-f'+str(i)][0:splitIndex].mean()
    h_bottom = df['y-f'+str(i)][splitIndex:].mean()
    for j in range(splitIndex):
        a.append(h_upper)
    for j in range(df.shape[0]-splitIndex):
        a.append(h_bottom)
    df['h'+str(i+1)] = a
    
df.head()


#  This is how the dataset looks after 2 iterations

# If we continue to iterate for 100 times , we can see the Loss of MSE(Fi) decreasing by a huge margin

for i in range(100):
    f = df.Salary.mean()
    if(i>0):
        df['f'+str(i)] = df['f'+str(i-1)] + df['h'+str(i)]
    else:
        df['f'+str(i)] = f
    df['y-f'+str(i)] = df.Salary - df['f'+str(i)]
    splitIndex = np.random.randint(0,df.shape[0]-1)
    a= []
    h_upper = df['y-f'+str(i)][0:splitIndex].mean()
    h_bottom = df['y-f'+str(i)][splitIndex:].mean()
    for j in range(splitIndex):
        a.append(h_upper)
    for j in range(df.shape[0]-splitIndex):
        a.append(h_bottom)
    df['h'+str(i+1)] = a
    
df.head()


#  Following is the graph for Iteration 1 , 10 and 99
#  We can see the loss decreasing and the model adapting to the dataset as the iteration increases



plt.figure(figsize=(15,10))
plt.scatter(df.Years,df.Salary)
plt.plot(df.Years,df.f1,label = 'f1')
plt.plot(df.Years,df.f10,label = 'f10')
plt.plot(df.Years,df.f99,label = 'f99')
plt.legend()



