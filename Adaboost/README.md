# Adaboost

## Introduction
AdaBoost (Adaptive Boosting) is a very popular boosting technique that aims at combining multiple weak classifiers to build one strong classifier. The original AdaBoost paper was authored by Yoav Freund and Robert Schapire.

## What is Adaboost?
AdaBoost is short for Adaptive Boosting and is a very popular boosting technique which combines multiple “weak classifiers” into a single “strong classifier”. AdaBoost models belong to a class of ensemble machine learning models. From the literal meaning of the word ‘ensemble’, we can easily have much better intuition of how this model works. Ensemble models take the onus of combining different models and later produce an advanced/more accurate meta model. This meta model has comparatively high accuracy in terms of prediction as compared to their corresponding counterparts.

## What is Boosting?
Boosting is an ensemble modeling technique which attempts to build a strong classifier from the number of weak classifiers. It is done building a model by using weak models in series. Firstly, a model is built from the training data. Then the second model is built which tries to correct the errors present in the first model. This procedure is continued and models are added until either the complete training data set is predicted correctly or the maximum number of models are added.

## Adaboost Algorithm
AdaBoost algorithm falls under ensemble boosting techniques, as discussed it combines multiple models to produce more accurate results and this is done in two phases:
<ol type="1">
  <li> Multiple weak learners are allowed to learn on training data. </li>
  <li> Combining these models to generate a meta-model, this meta-model aims to resolve the errors as performed by the individual weak learners. </li>
</ol>
