# XGBoost

XGBoost is an open-source library uses for the implementation of *gradient boosted trees* designed for speed and performance. It is based on *function approximation* by optimizing specific loss function as well as applying several regularization techniques. It also provides a parallel tree boosting to solve many data science problems. The same code runs on a major of a distributed environment and can solve millions of problems.

## History

The XGBoost was created by  **[Tianqi Chen](https://tqchen.com/)**, with many people contributing. It belongs to the umbrella of the **[Distributed Machine Learning Community]** or **[DMLC](https://dmlc.github.io/)**, who are also the creators of the popular mxnet deep learning library.

## Working of XGBoost

- **Boosting** **Trees** - In a normal machine learning model, like a decision tree, we would simply train a single model on our dataset and then further use it for prediction. We might work with arguments or apply certain new techniques, but in the end, we are still using a single model.
**Boosting** on the other hand uses an iterative approach, i.e. many models are combined to make a new one.

These models are not trained in isolation and the models are boosted one by one with each new model trained to correct the errors of the previous one. Models are added again and again until no further improvement can be made.

- **Gradient** **Boosting** - In this approach, new models are trained to predict the error of prior models.

<p align="center">
  <img width="460" height="400" src="https://1.bp.blogspot.com/-Ot1OxI24P0w/XPOAzrB_VnI/AAAAAAAAcj4/quBQ9FgK30gp2r-a7VrQMR1V5M8LPk7GACLcBGAs/s1600/xgboost.png">
</p>

## XGBoost Features

This library has been made in keeping the perspective of model training and execution in mind. It has prominent features as follows:

- High computational speed.
- High model performance.

## Interfaces supported

It supports the following interfaces:

- Command Line Interface(CLI)
- C++
- Python
- R
- Julia
- Java and JVM languages like Scala.

## Cons of XGBoost

- Slow - For a large dataset, it requires a large amount of time.
- Does not perform well in overlapped class.
- The selection of an appropriate kernel is tricky.

## Applications of XGBoost

It can be used to solve various kinds of problems in an efficient way, Such as:

- Regression problems 
- Classification and ranking problems
- User-defined prediction problems

## Resources

Here are some of the resources that you can look out for further reference:

- [Library introduction by author, Tianqi Chen](https://www.youtube.com/watch?v=ufHo8vbk6g4)
- [History of XGBoost by author, Tianqi Chen](https://sites.google.com/site/nttrungmtwiki/home/it/data-science---python/xgboost/story-and-lessons-behind-the-evolution-of-xgboost)
- [Brief knowledge of Gradient Boosting Machine Learning( XGBoost ) ](https://www.youtube.com/watch?v=wPqtzj5VZus)
