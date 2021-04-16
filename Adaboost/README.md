# Adaboost

## Introduction
AdaBoost (Adaptive Boosting) is a very popular boosting technique that aims at combining multiple weak classifiers to build one strong classifier. The original AdaBoost paper was authored by Yoav Freund and Robert Schapire.

## What is Adaboost?
AdaBoost is short for Adaptive Boosting and is a very popular boosting technique which combines multiple “weak classifiers” into a single “strong classifier”. AdaBoost models belong to a class of ensemble machine learning models. From the literal meaning of the word ‘ensemble’, we can easily have much better intuition of how this model works. Ensemble models take the onus of combining different models and later produce an advanced/more accurate meta model. This meta model has comparatively high accuracy in terms of prediction as compared to their corresponding counterparts.
AdaBoost algorithm falls under ensemble boosting techniques, as discussed it combines multiple models to produce more accurate results and this is done in two phases:
<ol type="1">
  <li> Multiple weak learners are allowed to learn on training data. </li>
  <li> Combining these models to generate a meta-model, this meta-model aims to resolve the errors as performed by the individual weak learners. </li>
</ol>

## What is Boosting?
Boosting is an ensemble modeling technique which attempts to build a strong classifier from the number of weak classifiers. It is done building a model by using weak models in series. Firstly, a model is built from the training data. Then the second model is built which tries to correct the errors present in the first model. This procedure is continued and models are added until either the complete training data set is predicted correctly or the maximum number of models are added.

## Working of Adaboost Algorithm
**Step 1:** A weak classifier (e.g. a decision stump) is made on top of the training data based on the weighted samples. Here, the weights of each sample indicate how important it is to be correctly classified. Initially, for the first stump, we give all the samples equal weights.

**Step 2:** We create a decision stump for each variable and see how well each stump classifies samples to their target classes. For example, in the diagram below we check for Age, Eating Junk Food, and Exercise. We'd look at how many samples are correctly or incorrectly classified as Fit or Unfit for each individual stump.

**Step 3:** More weight is assigned to the incorrectly classified samples so that they're classified correctly in the next decision stump. Weight is also assigned to each classifier based on the accuracy of the classifier, which means high accuracy = high weight!

**Step 4:** Reiterate from Step 2 until all the data points have been correctly classified, or the maximum iteration level has been reached.

**Fully grown decision tree (left) vs three decision stumps (right)**
![Image1](https://lh3.googleusercontent.com/kpQjxgGIxnSnMm495bDs0OZf4rE08E58PV1wwK9q10b_pL5AtKkRcY0OY5Hc_NFY0aW6iRQYAQDKuueEwnOfcEz9_IYyO-Ej-HwAqoFS_rQ779mP5HTHPKCy4x-lBmr33dd-Nw)

![Image2](https://blog.paperspace.com/content/images/2019/12/WhatsApp-Image-2019-12-30-at-11.55.02-AM.jpeg)
