# Linear Discriminant Analysis

#### A Thorough Walk Through

\
In the world of Machine Learning, we can distinguish two main areas: Supervised and unsupervised learning. The main difference between both lies in the nature of the data as well as the approaches used to deal with it. Dimensionality reduction is an approach to reduce the dimensions of data features thereby avoiding overfitting.

The main goal of dimensionality reduction techinques is to reduce the dimensions by removing the reduntant and dependent features by transforming the features from higher dimensional space to a space with lower dimensions.

There are 3 major types of dimensionality reduction :

1. Principal Component Analysis (PCA)
2. Linear Discriminant Analysis (LDA)
3. Generalized Discriminant Analysis (GDA)

**We will discuss about LDA here**

\
<space>

## Table of Content

| Sr. No. | Content                                      |
| ------- | -------------------------------------------- |
| 1       | Introduction to Linear Discriminant Analysis |
| 2       | Working of Linear Discriminant Analysis      |
| 3       | Steps to use in Linear Discriminant Analysis |
| 4       | Extension of Linear Discriminant Analysis    |
| 5       | Application of Linear Discriminant Analysis  |

\
<space>

## What is Linear Discriminant Analysis ?

\
<space>

> Linear Discriminant Analysis (LDA) is a dimensionality reduction technique. As the name implies dimensionality reduction techniques reduce the number of dimensions (i.e. variables) in a dataset while retaining as much information as possible.

Linear Discriminant Analysis is a supervised classification technique which takes labels into consideration.This category of dimensionality reduction is used in biometrics,bioinformatics and chemistry.

\
<space>

## How does Linear Discriminant Analysis Work ?

The goal of Linear Discriminant Analysis is to project the features in higher dimension space onto a lower dimensional space.

For example, lert's say that we plotted the relationship between two variables where each color represent a different class :

<p align="center">
<img width="400" height="400" src="https://miro.medium.com/max/875/1*o2TKovc_lkJ9_ISxZxrbog.png">
</p>

So above we have dummy data plotted on a 2D space. Let's say we want to reduce the number of dimensions to one. The approach is:

<p align="center">
<img width="400" height="400" src="https://miro.medium.com/max/875/1*5lugB_AavKEr3ghDGC6hOA.png">
</p>

<p align="center">
<img width="500" height="100" src="https://miro.medium.com/max/794/1*Z202fIHoHkW5KhxxXcQ8jA.png">
</p>

LDA, uses the information from both features to create a new axis and projects the data on to the new axis in such a way as to minimizes the variance and maximizes the distance between the means of the two classes.
This can be depicted using the following images :

<p align="center">
<img width="400" height="400" src="https://miro.medium.com/max/875/1*Fz3JQ80No5Nnbap28EGRTg.png">
</p>

<p align="center">
<img width="400" height="400" src="https://miro.medium.com/max/875/1*5lhckC2RQzq28zNL7WtU5A.png">
</p>

<p align="center">
<img width="500" height="100" src="https://miro.medium.com/max/875/1*W48aQ0LkZ5dm1_uow6FD2w.png">
</p>

\
<space>

## Steps to use LDA

\
<space>

LDA is achieved using the following 3 steps:

1. Calculate the separability between different classes(i.e the distance between the mean of different classes) also called as between-class variance.
   ![1st step](https://miro.medium.com/max/389/1*hcWeLRuL5qwOwUt7DNxd_g.png)
2. Calculate the distance between the mean and sample of each class,which is called the within class variance.
   ![2nd step](https://miro.medium.com/max/665/1*eB-JPJD6s6yc734WmR6Ivw.png)
3. Construct the lower dimensional space which maximizes the between class variance and minimizes the within class variance.Here let "P" be the lower dimensional space projection,which is called **Fisher’s criterion**.

   ![3rd step](https://miro.medium.com/max/340/1*XeWA6T-swOZGUMmfrLMIIA.png)

\
<space>

## Preparing Data For LDA

Here are some ways in which you can create data :

- **Classification Problems**: As LDA is intended for classification problems where the output variable is categorical. LDA supports both binary and multi-class classification.
- **Gaussian Distribution**: The standard implementation of the model assumes a Gaussian distribution of the input variables. Assume reviewing the univariate distributions of each attribute and using transforms to make them more Gaussian-looking (eg: log and root for exponential distributions). More about Gaussian Mixture Model [here](https://github.com/Algo-Phantoms/Algo-ScriptML/blob/main/Gaussian%20Mixture%20Model/README.md).
- **Remove Outliers**: Outliers usually hinders our expected result as it talks about extremity (too small, too large) so it's better to remove it. These can skew the basic statistics used to separate classes in LDA such the mean and the standard deviation.
- **Same Variance**: LDA assumes that each input variable has the same variance. So it's almost always a good idea to standardize your data before using LDA so that it has a mean(&micro;) of 0 and a standard deviation(&sigma;) of 1.

\
<space>

## Extension of LDA

LDA has several other variations and some of them are listed below:

1. **Regularized Discriminant Analysis (RDA)** : Here, the variance (or covariance) is subjected to regularization which helps in moderating influence of different variables on LDA.
2. **Quadratic Discriminant Analysis (QDA)** : Here, each class uses its own variance/covariance or covariance of multiple input variables for estimation.
3. **Flexible Discriminant Analysis (FDA)** : This is quite flexible method and it uses splines when the inputs are non-linear combinations of data.

\
<space>

## Applications of LDA

- **Medical** : Patient’s diseases can also be classified as moderate, mild or severe using linear discriminant analysis. Such classification plots the medical treatment to the various patient parameters to make a distinction between the classes. This helps doctors to tackle the disease.
- **Identification of customers** : In the marketing world, it is important to find the right customer for a product. So A survey of customers wishing to buy a particular product at a specific shopping mall can easily provide custom features which can then be classified using Linear discriminant analysis to help select and identify the customer features.
- **Facial Recognition** : In the modern computer era, the human face is represented by several points with pixel values by linear discriminant analysis in machine learning. Linear discriminant analysis (LDA) is used here to reduce the number of features to a more manageable number before the process of classification. Each of the new dimensions generated is a linear combination of pixel values, which form a template. These combinations are called Fisher faces and the dimension as the Fisher’s linear discriminant.

\
<space>

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
2. [Towards Data Science](https://towardsdatascience.com/)
3. [Sebastian Raschka](https://sebastianraschka.com/Articles/2014_python_lda.html)
4. [StatQuest](https://www.youtube.com/watch?v=azXCzI57Yfc)
