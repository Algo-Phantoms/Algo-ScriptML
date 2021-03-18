
Using the script

## import the module

>import linearRegression
>
>linearReg_model = LinearRegression()

## train the data

>linearReg_model.fit(x_train, y_train)

## predict the model

>y_pred = linearReg_model.predict(x_test)
 ## Regression:

Firstly let’s see what's regression.  Regression is a technique for predicting a goal value using independent predictors. This method is primarily used for forecasting and determining cause and effect relationships among variables. The number of independent variables and the form of relationship between the independent and dependent variables is the key points that cause the differences in regression techniques.

## Linear regression

One of the most fundamental and commonly used Machine Learning algorithms is linear regression. It's a statistical methodology for conducting the predictive analysis. The linear regression algorithm shows a linear relationship between a dependent (y) variable and one or more independent (y) variables, hence the name. Since linear regression reveals a linear relationship, it determines how the value of the dependent variable changes as the value of the independent variable changes.

![](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-in-machine-learning.png)

Linear regression is mathematically represented as:-

y=a0 +a1.x

Here,
y= Dependent variable
a0= Intercept of line
a1= Linear regression coefficient
x= Independent variable

There are two types of linear regression:-

Simple linear regression- It is a is Linear Regression algorithm that uses a single independent variable to predict the value of a numerical dependent variable.

Multiple linear regression- It is a  Linear Regression algorithm that uses more than one independent variable to estimate the value of a numerical dependent variable.

## Cost function(J):

When using linear regression, our main aim is to find the best fit line, which means that the difference between expected and actual values should be as small as possible. The line with the best fit would have the least amount of error. The cost function assists us in deciding the best possible values for a0 and a1 in order to achieve the best possible fit line for the data points. Since we want the best values for a0 and a1, we transform this into a minimization problem in which we want to minimize the difference between the expected and actual values.
The cost function can be used to determine the accuracy of a mapping function that maps an input variable to an output variable. The hypothesis function is another name for the mapping function. The error discrepancy is determined by the difference between expected and ground truth values. We square the error difference, add all of the data points together, and divide the total number of data points by two. This gives you the average squared error for all of your data points. As a consequence, the Mean Squared Error(MSE) function is another name for this cost function.

![](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-in-machine-learning4.png)


Here, 
N= total no. of observation
yi= actual value 
a1xi+a0=predicted value

## Gradient Descent:

Gradient descent is a method of reducing the cost function by modifying a0 and a1 (MSE). The idea is that we start with some a0 and a1 values and then reduce the cost by adjusting them iteratively. Gradient descent assists us in changing the values. The gradient always points in the direction of the steepest loss function rise. In order to minimize loss as quickly as possible, the gradient descent algorithm takes a step in the direction of the negative gradient. The learning rate in the gradient descent algorithm is the number of steps you take. This dictates how easily the algorithm reaches the minima.
A smaller learning rate will get you closer to the minima, but it will take longer to achieve it; a larger learning rate converges faster, but there is a risk of overshooting the minima.

![](https://miro.medium.com/max/470/1*D4Q7zeRBmZ3z1CbD37CIhg.png)

The partial derivates are the gradient descent and are used to update the value of a0 and a1

For more clear perspective you can also go through the following video:
https://www.youtube.com/watch?v=E5RjzSK0fvY
