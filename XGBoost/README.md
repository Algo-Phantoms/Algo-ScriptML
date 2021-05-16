## Boosting:

Boosting is yet another ensemble machine learning technique which is designed to improve or boost the performance of a no. of weak learning models working together.

The weak Classifier:

A weak classifier is a model which can’t produce a sable result on it’s own but performs better than a random guess.

Here is the Mathematical representation of a boosting classifier<br>
                                               ![formula](https://user-images.githubusercontent.com/58946705/111885101-a7c5a580-89eb-11eb-8dee-fd71bc899a2d.jpg)

Here, **T** is the no. of classifiers, **ft** is a weak learner and **x** is the data input. It returns the class

 It can be depicted by the below diagram:
 ![flowchart](https://user-images.githubusercontent.com/58946705/111885187-2a4e6500-89ec-11eb-919d-a9f260f2487c.jpg)


## XGBoost
XGBoost improves the gradient boosting method even further.
It was developed by Tianqi Chen in C++ but now has interfaces for Python, R, Julia.

XGBoost's objective function is the sum of loss function evaluated over all the predictions and a regularisation function for all predictors ( 𝑗  trees). In the formula  𝑓𝑗  means a prediction coming from the  𝑗𝑡ℎ  tree.


![xgbf](https://user-images.githubusercontent.com/58946705/111885430-be6cfc00-89ed-11eb-825f-7d769a428a68.jpg)



Unlike the other tree-building algorithms, XGBoost doesn’t use entropy or Gini indices. Instead, it utilises gradient (the error term) and hessian for creating the trees.


### Understanding the Algorithm:
The working of XGBoost is smart as well as complex. But if we break it down into simple steps, it is quite easy and interesting to understand. So we can divide the overall working of this algorithm into 5 main steps, as follows:

**Step 1**
First we create a base model, and produce the predictions out of it (the ŷ value).
Note: We can take a random value as prediction to work upon, in xgboost, it is 0.5.

**Step 2**
Now we calculate the residuals from the prediction we just made.
Residuals can be calculated as the difference between the actual value ( y ) and the predicted value ( ŷ ), that will be y - ŷ.

**Step 3**
Fit an XGBoost Tree. <br>
This is the main step which involves construction of weak learner, Regularization and Pruning of the xgb tree : <br>
•	We start with a leaf, which contains the residuals and perform a split for a feature in binary fashion. <br>
•	Then we calculate a measure called ‘Similarity score’ for that node which can be calculated from this formula : <br> ** (Sum of residuals)^2 / no. of residuals + λ ** <br>
•	Now we decide if we can improve our XGBoost tree by taking different thresholds for splitting the tree for a feature (for eg. the intermediate average of data points).
We compare and decode the performance of a split by calculating another measure called ‘Gain’. <br>
•	Gain can be calculated as the difference between the Similarity score of a branch after split and the Similarity Score before the split. We can now compare this calculation to choose the maximum gain proving threshold for split. <br>
•	Now we perform Pruning of the tree by using a Gamma Value ‘γ’ supplied by the user. The pruning of a branch is done by comparing the Gain and the Gamma value γ. <br>
•	If  γ – Gain is positive, we do not prune the branch (we include it) <br>
•	If γ – Gain is negative, we prune the branch ( exclude it) <br>

**Step 4**
Now we calculate the output value from the leaf nodes of the tree from the following formula:
 Output = Sum Of Residulas / total no. of residuals + λ
 
 **Step 5**
 Make Predictions: <br>
 For a new data instance, the prediction is calculated by first   traversing through the XGB Tree we built, and finding the leaf node to which it belongs. 
Now we calculate the output value from that leaf and calculate the new predictions. <br>
**new prediction = previous prediction + L.R. X Output value** <br>
where L.R. is the Learning Rate as supplied by the user.


### Advantages : 
Following are the advantages of XGBoost : 
•	Parallelization : It supports parallelization, i.e. it can make use of all the cores of the system upon which it its working, so as to maximize its performance <br>
•	Cache Optimization : It holds all the intermediate calculations and statistics in the cache memory of the system which can be fetched and worked upon quickly <br>
•	Out of memory computation : It also supports out of memory computation, i.e. if the systems goes out of memory, it can still handle and perform the instructions in it’s distributed version. <br>
•	Regularization : This is a very important feature of XGBoost, it can control the variance and overfitting of the model through it’s regularization parameter  ‘λ’. <br>
•	Auto-pruning : This is yet another feature of XGBoost which contributes towards the overall generalization and overfitting control of the trees. <br>
•	Missing Value treatment : It is also capable of handeling missing values in the dataset. <br>
•	Other features : It’s Multi- language support , portability and integrability also makes it the choice. <br>

### Disadvantages :
•	Requires heavy computation <br>
•	It has a Black Box Nature <br>
•	More Likely to overfit than Bagging techniques <br>
•	Training time is high <br>

