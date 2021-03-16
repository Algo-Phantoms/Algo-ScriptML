# Logistic Regression Implementation Base

### Parameters passed to the functions :

**X** - array containing all the features

**y** - array containing the classification values

**theta** - row vector contaning weights

**alpha** - variable for denoting the learning rate

**num_itr** - number of iterations

### Functions Used :

  i)  **normalize_features** : function to normalize the features present in array X. 
  
  ii) **hypothetical_function** : function that takes the array X and weight theta and returns the logistic function. X_temp is the array that contains the intercept term.
  
  iii)**compute_cost** : calculates the cost of the entire dataset. 
  
  iv) **gradient_descent** : function to implement the gradient descent to get optimal weights. delta is the gradient array.
  
  v)  **logistic_regression** : function that trains the model
  
  vi) **calculate_y_predicted** : function to test the model accuracy.
