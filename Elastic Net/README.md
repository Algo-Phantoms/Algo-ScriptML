# 🕸️ ELASTIC NET REGRESSION

## Introduction

Elastic net linear regression uses the penalties from both the lasso and ridge techniques to regularize regression models. The technique combines both the lasso and ridge regression methods by learning from their shortcomings to improve on the regularization of statistical models.

<img src="https://cdn.corporatefinanceinstitute.com/assets/elastic-net1-1200x753.png" width="600" height="400"/>

The elastic net method performs variable selection and regularization simultaneously. This technique is most appropriate where the dimensional data is greater than the number of samples used. Groupings and variables selection are the key roles of the elastic net technique.

The modified cost function for Elastic-Net Regression is given below:

<img src="https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-b9ecaab808c8021fe133006037b4c435_l3.svg" width="800" height="400"/>

Here, w(j) represents the weight for jth feature.  

n is the number of features in the dataset.

lambda1 is the regularization strength for L-1 norm.

lambda2 is the regularization strength for L-2 norm.

## Advantages

▪ Doesn’t have the problem of selecting more than n predictors when n<<p, whereas LASSO saturates when n<<p.

▪ Encourages grouping effect in the presence of highly corelated predictors.

## Disadvantages

▪ Computationally more expensive than LASSO or Ridge.

▪ Naive Elastic Net suffers from double shrinkage.

## References

▪ https://corporatefinanceinstitute.com/resources/knowledge/other/elastic-net/

▪ https://medium.com/@gokul.elumalai05/pros-and-cons-of-common-machine-learning-algorithms-45e05423264f

▪ https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/

▪ https://www.slideshare.net/ShangxuanZhang/ridge-regression-lasso-and-elastic-net
