# K Nearest Neighbors Algorithm 
# Introduction 
The K-nearest neighbors is a simple and easy-to-implement supervised machine learning algorithm. It can be used to solve both classification as well as regression problems. This algorithm assumes that similar data points are close to each other in the scatter plot.  

# Choosing the right value for K 
The best way to decide this is by trying out several values of K (number of nearest neighbors) before settling on one. Low values of K (like K = 1 or K = 2) can be noisy and subject to outliers. If we take large values of K, a category with only a few values in it will always be voted out by other categories. Choose the value of K that reduces the number of errors. We usually make K an odd number to have a tiebreaker.    
 
# Algorithm  
1. Start with a dataset with known categories.<br>
2. Initialize K(number of nearest neighbours).<br>
3. For each data point in the training data:<br>
   3.1 calculate the distance between the data point and the current example from the data.<br>
   3.2 add the distance and the index of the example to an ordered collection.<br>
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances<br>
5. Pick the first K entries from the sorted collection.<br>
6. Get the labels of the selected K entries.<br>
7. If regression, return the mean of the K labels. (return average of K labels)<br>
8. If classification, return the mode of the K labels. (return mode of K labels)<br>

<p align="center">
  <img width="460" height="300" src="https://www.edureka.co/blog/wp-content/uploads/2018/07/KNN-Algorithm-k3-edureka-437x300.png">
</p>

# Advantages
1. KNN algorithm is very easy to implement. It requires only two parameters to implement i.e. the value of K and the distance function.<br>
2. There is no need to build a model, tune parameters, or make additional assumptions.<br>
3. New data can be added seamlessly which will not impact the accuracy of the algorithm.<br>
# Disadvantages
1. KNN algorithm does not work well with large datasets. In large datasets, the cost of calculating the distance between the new point and the existing points is huge which degrades the performance of the algorithm. <br>
2. It is sensitive to noisy data, missing values and outliers. <br>
# References
1. https://www.youtube.com/watch?v=HVXime0nQeI <br>
2. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761 <br>
