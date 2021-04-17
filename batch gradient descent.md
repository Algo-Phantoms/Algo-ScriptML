# Batch gradient descent

-Before starting off with gradient descent let us look at what is optimization.
# Optimization
-In lyman words,we need to find the global minimum of the objective function .This is the feasible region in which the global minimum .This is feasible if the objective function is convex,i.e.,any local minimum is a global minimum.

-find the lowest possible value of the objective function within its neighbourhood.That's usually the case if the objective function is not convex as the case wtih most deep learning problems.

-make sure to scale the data if the values are on very different scales.If we don't scale the data, the level curves (contours) would be narrower and taller which means it would take longer time to converge.
# What is batch gradient descent?
-Batch gradient descent is an optimization technique to find the efffective parameters in which we sum up overall examples on each iteration when performing the updates to the parameeters.Therefore,for each update we have to sum overall examples.
w=w-(learning rate)*(cost)
# The main advantages
We can use fixed learning rate during training without worrying about learning rate decay.
It has straight trajectory towards the minimum and its guraanteed to converge in theory to the global minimum if the loss function is convex.and to a local minimum if the loss function is not convex. 

It has unbiased estimate of gradients.The more the examples,the lower the standard error.
# The main disadvantages
-Even though we can use vectorized implementation,it may still be slow to go over all examples especially when we have large datasets.
-Each steps of learning happens after going through all examples and hence maybe redundant and don't contribute much to update.







-











