## RANDOM FOREST 

### Introduction 
Random forest is a popular machine learning algorithm that belongs to the supervised learning technique.
It can be used for both classification and regression and is based on the concept of ensemble learning i.e. a process of
combining multiple classifiers to solve a complex problem and to improve the performance of the model.
Random forest makes use of the bagging technique.

### What is the bagging technique?
Suppose you have a dataset D which has 'n' records and you have to make predictions on it. You'll make multiple base models say M1, M2, M3,...Mn.
Now, we will choose a subset of the records from D, say 'm' records (dm1) and train model M1 on it. Again resample the dataset and choose some 'm' records (dm2)
and train model M2 on it. This is called as row sampling with replacement. Keep doing this for every model from M1 to Mn.\
Note that here 'm' will always be less than 'n' and while choosing the subset of data some records may or may not be repeated. \
Now, we take some test data Test_D and give it to each model M1, M2,...Mn and get results as R1, R2,...Rn respectively and 
the final result will be the vote which is in majority.

> Bagging is also called as bootstrap aggregation where bootstrap means the resampling technique and aggregation refers to the part where we are combining the results
of all the models to get the final result.

### How Random Forest algorithm works?
In random forest, the models M1, M2...etc. are nothing but decision trees.
There are two phases namely:
  1. Creating the random forest by combining N decision trees
  2. Making predictions for each tree

#### Algorithm:
1. Starts by selecting random samples from given dataset using the bootstrap technique.
2. This algorithm will construct a decision tree for every sample, then it will get prediction result from each decision tree.
3. Next, voting will be performed for every predicted result.
4. At last, select the most voted prediction as the final prediction result.

<p align="center">
    <img src="https://editor.analyticsvidhya.com/uploads/74060RF%20image.jpg" width="750" height="500">
</p>

NOTE: While using the bagging technique here, some rows and features will be chosen & not only rows. Every sample will have some
rows and features which may or may not be repeated from the previous samples.

### Classifier and Regressor
- For a Random Forest Classifier --> the final result will be the majority vote.
- For a Random Forest Regressor --> the final result will be mean of results of all the decision trees.

 ### Advantages
- Both classification and regression tasks
- Handle the missing values and maintains accuracy
- Prevents overfitting
- Handle large dataset with higher dimensionality

### Disadvantages
- Not much suitable for regression
- You have very little control on what the model does

### Applications
- Finance sector : 
  1) To determine a stock's future behaviour
  2) Credit card fault detection
- Medicine :
  1) For identifying correct combination of components in a medicine
  2) To analye a patient's medical history to identify diseases 
- E-commerce : 
  1) Product recommendation
  2) Price optimization

### References
- https://builtin.com/data-science/random-forest-algorithm
- https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674
- https://www.youtube.com/watch?v=D_2LkhMJcfY
- https://www.youtube.com/watch?v=nxFG5xdpDto
