# Naive Bayes

## Introduction
Naive Bayes is a poweful classification algorithim which is based on Bayes's Theorem . It  assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. So when a model has a bad performance, the reason leading to that may be the dependence between predictors.

[![Naive-Bayes.jpg](https://i.postimg.cc/dthc5PSp/Naive-Bayes.jpg)](https://postimg.cc/nCfWL5xG)

## Types of Naive Bayes Classifier:

 - Multinomial Naive Bayes -used for discrete counts eg -document classification problem
 - Bernoulli Naive Bayes -The binomial model is useful if your feature vectors are binary (i.e. zeros and ones).
 - Gaussian Naive Bayes - It assumes that features follow a normal distribution.

## Advantages 

 - Highly scalable and fast algorithm.   
 - It is not prone to overfitting.
 - An excellent choice for Text Classification problems.
 - It can be easily trained on a small dataset.
 - Perform well in multi class prediction

## Disadvantages

-   According to the “Zero Conditional Probability Problem.”, if a given feature and class have frequency 0, then the conditional probability estimate for that category comes out as 0. This problem is cumbersome as it wipes out all the information in other probabilities too.
-  Its strong assumption of independence class features. It is nearly impossible to find such data sets in real life.

## Applications of Naive Bayes Algorithm

1.  **Sentiment Analysis**.
2.  **Spam filtration**:  Several server-side email filters, such as SpamBayes, SpamAssassin, and Bogofilter, make use of this technique.
3.  **Text classification**: Used as a probabilistic learning method for text classification. 
4.  **Recommendation System**.

## References

https://web.stanford.edu/~jurafsky/slp3/4.pdf
https://web.stanford.edu/class/archive/cs/cs109/cs109.1176/lectures/22-NaiveBayes.pdf
https://www.youtube.com/watch?v=nt63k3bfXS0
https://www.youtube.com/watch?v=IVKF_wmIdiI
