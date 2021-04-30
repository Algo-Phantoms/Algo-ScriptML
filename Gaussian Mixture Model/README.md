# Gaussian Mixture Models

#### A Thorough Walk Through

\
In the world of Machine Learning, we can distinguish two main areas: Supervised and unsupervised learning. The main difference between both lies in the nature of the data as well as the approaches used to deal with it. Clustering is an unsupervised learning problem where we intend to find clusters of points in our dataset that share some common characteristics.

Clustering is a concept we typically learn early on in our machine learning journey and it’s simple enough to grasp. I’m sure you’ve come across or even worked on projects like customer segmentation, market basket analysis, etc.
_An Example of Clustering:_

![Clustering Image](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/0.png)

But here's a catch, clustering has many layers. It isn’t limited to the basic algorithms. It is a powerful unsupervised learning technique that we can use in the real-world with unerring accuracy.

> **Gaussian Models are one such algorithm**

\
<space>

## Table of Content

| Sr. No. | Content                                             |
| ------- | --------------------------------------------------- |
| 1       | Introduction to Gaussian Mixture Models             |
| 2       | The Gaussian Distribution                           |
| 3       | What is Expectation-Maximization?                   |
| 4       | Expectation-Maximization in Gaussian Mixture Models |
| 5       | Expectation Maximization in General                 |

\
<space>

## Introduction to Gaussian Mixture Models (GMMs)

> _A Gaussian Mixture is a function that is_ > _comprised of several Gaussians, each identified by k ∈ {1,…, K}, where K is the number of clusters of our dataset._

Each Gaussian k in the mixture is comprised of the following parameters:

- A mean &micro; that defines its centre.
- A covariance &sum; that defines its width. This would be equivalent to the dimensions of an ellipsoid in a multivariate scenario.
- A mixing probability &pi; that defines how big or small the Gaussian function will be.

_(More on this in the next Section)_

Assume we have 3 Gaussian distributions. These have a certain mean (&micro;<sub>1</sub>, &micro;<sub>2</sub>, &micro;<sub>3</sub>) and variance (&sigma;<sub>1</sub>, &sigma;<sub>2</sub>, &sigma;<sub>3</sub>) value respectively. For a given set of data points, our GMM would identify the probability of each data point belonging to each of these distributions.

_What ? Probability ?_

**Yes, Gaussian Mixture Models are probabilistic models and use the soft clustering approach for distributing the points in different clusters.**

_eg:_ \
Say we have 3 clusters that are denoted by three colors \-- Blue, Green, and Cyan. Let’s take the data point highlighted in red. The probability of this point being a part of the blue cluster is 1, while the probability of it being a part of the green or cyan clusters is 0.

![Cluster1](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/Screenshot-from-2019-10-21-12-52-06-670x440.png)

\
Now, consider another point – somewhere in between the blue and cyan (highlighted in the below figure). The probability that this point is a part of cluster green is 0, right? And the probability that this belongs to blue and cyan is 0.2 and 0.8 respectively.

![Cluster2](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/Screenshot-from-2019-10-21-12-53-29.png)

\
Gaussian Mixture Models use the soft clustering technique for assigning data points to Gaussian distributions. I’m sure you’re wondering what these distributions are so let me explain that in the next section.

\
<space>

## The Gaussian Distribution

Some of us might have studied the Gaussian Distribution (or Normal Distribution). It has a bell-shaped curve, with the data points symmetrically distributed around the mean value.

The below image has a few Gaussian distributions with a difference in mean &micro; and variance &sigma;<sup>2</sup>. Remember that the higher the σ value more would be the spread:

![Gaussian Distribution](https://miro.medium.com/max/875/1*lTv7e4Cdlp738X_WFZyZHA.png)

\
Here, we can see that there are three Gaussian functions, hence K = 3. Each Gaussian explains the data contained in each of the three clusters available.

Here the Gaussian function is given by:

![Gaussian function](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/pdf_var-1.png)\
where &micro; is the mean and &sigma;<sup>2</sup> is the variance.

A 3D Bell Curve can be shown as:

![Bell Curve 3D](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/gaussians-3d.png)

Here the probability density function would be:

![pdf](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/pdf_var-2.png)

where x is the input vector, &micro; is the 2D mean vector, and &sum; is the 2×2 covariance matrix. The covariance would now define the shape of this curve. We can generalize the same for d-dimensions.

> _Thus, this multivariate Gaussian model would have x and &micro; as vectors of length d, and &sum; would be a dxd covariance matrix._

\
<space>
These values are determined using a technique called Expectation-Maximization (EM).

\
<space>

## What is Expectation-Maximization?

\
<space>

> _The expectation-maximization algorithm is an approach for performing maximum likelihood estimation in the presence of latent (missing) variables._

\
**Since we do not have the values for the latent variables, Expectation-Maximization tries to use the existing data to determine the optimum values for these variables and then finds the model parameters.** Based on these model parameters, we go back and update the values for the latent variable, and so on.

Broadly, the Expectation-Maximization algorithm has two steps:

- **(Expectation)** Assign each data point to a cluster probabilistically. In this case, we compute the probability it came from the red cluster and the yellow cluster respectively.
- **(Maximization)** Update the parameters for each cluster (weighted mean location and variance-covariance matrix) based on the points in the cluster (weighted by their probability assigned in the first step).

Expectation-Maximization is the base of many algorithms, including Gaussian Mixture Models.

## Expectation-Maximization in Gaussian Mixture Models

Let’s understand this using another example. I want you to visualize the idea in your mind as you read along. This will help you better understand what we’re talking about.

We should follow the following steps for the solution:

1. Start with an arbitrary initial choice of parameters.
2. Apply Expectation
3. Apply Maximization
4. Repeat steps 2 and 3 to convergence.

Let the parameters of our model be

![](https://miro.medium.com/max/720/1*rVAJtuaMXEgcdklaSBRzQw.png)

Let’s say we need to assign $k$ number of clusters. This means that there are k Gaussian distributions, with the mean and covariance values to be &micro;<sub>1</sub>, &micro;<sub>2</sub>, .. &micro;<sub>k</sub> and &sum;<sub>1</sub>, &sum;<sub>2</sub>, .. &sum;<sub>k</sub> . Additionally, there is another parameter for the distribution that defines the number of points for the distribution. Or in other words, the density of the distribution is represented with &Pi;<sub>i</sub>.

**Step 1:**
\
<space>
Initialise θ accordingly. For instance, we can use the results obtained by a previous K-Means run as a good starting point for our algorithm.

**Step 2 (Expectation step):**

For each point x<sub>i</sub>, calculate the probability that it belongs to cluster/distribution c<sub>1</sub>, c<sub>2</sub>, … c<sub>k</sub>. This is done using the below formula:

\
<space>
![formula](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/probab_gmm_1.png)

\
<space>
The result of this operation would be high when the point is assigned to the right cluster and lower otherwise.

**Step 3 (Maximization step):**

Here we go back and update the &Pi;, &micro; and &sum; values. These are updated in the following manner:

1. The new density is defined by the ratio of the number of points in the cluster and the total number of points:  
    \
   <space>
   ![Pi update](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/mean-formula-gmm.png)

2. The mean and the covariance matrix are updated based on the values assigned to the distribution, in proportion with the probability values for the data point. Hence, a data point that has a higher probability of being a part of that distribution will contribute a larger portion:

   \
   <space>
   ![Mu and Sigma Update](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/10/formulas-gmm-e1571990956378-300x128.png)

\
 <space>

**Step 4 :**

Now based on the updated values generated from this step , we calculate the probabilities for each data points and update the values iteratively. This is done in order to maximize the [log-likelihood function](https://en.wikipedia.org/wiki/Likelihood_function#:~:text=Log%2Dlikelihood%20function%20is%20a,to%20maximizing%20the%20log%2Dlikelihood.).

In other words, _k-means only considers the mean to update the centroid while GMM takes into account the mean as well as the variance of the data._

> ![example of EM](https://miro.medium.com/max/450/0*kE-YHM1yJfxnbCLt.gif) \
>  <space> [Expectation Maximization for Old Faithful Eruption Data](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)

\
<space>

## Expectation Maximization in General

1. We have some data X of whatever form.
2. We posit there is also unobserved (latent) data Δ, again of whatever form.
3. We have a model with parameters &theta;.
4. We have the ability to compute the log-likelihood ℓ(θ; X, Δ). Specifically, the log of the probability of observing our data and specified assignments of the latent variables given the parameters.
5. We also have the ability to use the model to compute the conditional distribution Δ|X given a set of parameters. We will denote this P(Δ|X; θ).
6. Consequently we can compute the log-likelihood ℓ(θ; X). This is the log of the probability of observing our data given the parameters (without assuming an assignment for the latent variables).

_More about log-likelihood [here](https://en.wikipedia.org/wiki/Likelihood_function#:~:text=Log%2Dlikelihood%20function%20is%20a,to%20maximizing%20the%20log%2Dlikelihood.)_

\
<space>
**References :**

1. [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)

2. [Scikit Learn](https://scikit-learn.org/stable/modules/mixture.html#:~:text=A%20Gaussian%20mixture%20model%20is,Gaussian%20distributions%20with%20unknown%20parameters.)

3. [Wikipedia](https://en.wikipedia.org/wiki/Mixture_model)
